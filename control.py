# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Model-predictive non-linear control example.
"""

import collections

from jax import lax, grad, jacfwd, jacobian, vmap
import jax.numpy as jnp
import jax.ops as jo


# Specifies a general finite-horizon, time-varying control problem. Given cost
# function `c`, transition function `f`, and initial state `x0`, the goal is to
# compute:
#
#   argmin(lambda X, U: c(T, X[T]) + sum(c(t, X[t], U[t]) for t in range(T)))
#
# subject to the constraints that `X[0] == x0` and that:
#
#   all(X[t + 1] == f(X[t], U[t]) for t in range(T)) .
#
# The special case in which `c` is quadratic and `f` is linear is the
# linear-quadratic regulator (LQR) problem, and can be specified explicity
# further below.
#
ControlSpec = collections.namedtuple(
    'ControlSpec', 'cost dynamics horizon state_dim control_dim')


# Specifies a finite-horizon, time-varying LQR problem. Notation:
#
#   cost(t, x, u) = sum(
#       dot(x.T, Q[t], x) + dot(q[t], x) +
#       dot(u.T, R[t], u) + dot(r[t], u) +
#       dot(x.T, M[t], u)
#
#   dynamics(t, x, u) = dot(A[t], x) + dot(B[t], u)
#
LqrSpec = collections.namedtuple('LqrSpec', 'Q q R r M A B')


CILQRSpec = collections.namedtuple(
    'CILQRSpec', 'constraints ControlSpec mu')

dot = jnp.dot
mm = jnp.matmul


def mv(mat, vec):
  assert mat.ndim == 2
  assert vec.ndim == 1
  return dot(mat, vec)


LOOP_VIA_SCAN = False


def fori_loop(lo, hi, loop, init):
  if LOOP_VIA_SCAN:
    return scan_fori_loop(lo, hi, loop, init)
  else:
    return lax.fori_loop(lo, hi, loop, init)


def scan_fori_loop(lo, hi, loop, init):
  def scan_f(x, t):
    return loop(t, x), ()
  x, _ = lax.scan(scan_f, init, jnp.arange(lo, hi))
  return x


def trajectory(dynamics, U, x0):
  '''Unrolls `X[t+1] = dynamics(t, X[t], U[t])`, where `X[0] = x0`.'''
  T, _ = U.shape
  d, = x0.shape

  X = jnp.zeros((T + 1, d))
  X = jo.index_update(X, jo.index[0], x0)

  def loop(t, X):
    x = dynamics(t, X[t], U[t])
    X = jo.index_update(X, jo.index[t + 1], x)
    return X

  return fori_loop(0, T, loop, X)



def make_lqr_approx(p):
  """ Returns a function that uses first order approximation of nonlinear LQR 
      We linearize around different points but the gradients are constant """
  T = p.horizon
  def approx_timestep(t, x, u):
    M = jacfwd(grad(p.cost, argnums=2), argnums=1)(t, x, u).T
    Q = jacfwd(grad(p.cost, argnums=1), argnums=1)(t, x, u)
    R = jacfwd(grad(p.cost, argnums=2), argnums=2)(t, x, u)
    q, r = grad(p.cost, argnums=(1, 2))(t, x, u)
    A, B = jacobian(p.dynamics, argnums=(1, 2))(t, x, u)
    return Q, q, R, r, M, A, B

  _approx = vmap(approx_timestep)

  def approx(X, U):
    assert X.shape[0] == T + 1 and U.shape[0] == T
    U_pad = jnp.vstack((U, jnp.zeros((1,) + U.shape[1:])))
    Q, q, R, r, M, A, B = _approx(jnp.arange(T + 1), X, U_pad)
    return LqrSpec(Q, q, R[:T], r[:T], M[:T], A[:T], B[:T])

  return approx


def lqr_solve(spec):
  """ Given lqr spec, returns the K and k matrices used to calculate trajectory (backwards pass) """
  EPS = 1e-7
  T, control_dim, _ = spec.R.shape
  _, state_dim, _ = spec.Q.shape

  K = jnp.zeros((T, control_dim, state_dim))
  k = jnp.zeros((T, control_dim))

  def rev_loop(t_, state):
    t = T - t_ - 1
    spec, P, p, K, k = state

    Q, q = spec.Q[t], spec.q[t]
    R, r = spec.R[t], spec.r[t]
    M = spec.M[t]
    A, B = spec.A[t], spec.B[t]

    AtP = mm(A.T, P)
    BtP = mm(B.T, P)
    G = R + mm(BtP, B)
    H = mm(BtP, A) + M.T
    h = r + mv(B.T, p)
    K_ = -jnp.linalg.solve(G + EPS * jnp.eye(G.shape[0]), H)
    k_ = -jnp.linalg.solve(G + EPS * jnp.eye(G.shape[0]), h)
    P_ = Q + mm(AtP, A) + mm(K_.T, H)
    p_ = q + mv(A.T, p) + mv(K_.T, h)

    K = jo.index_update(K, jo.index[t], K_)
    k = jo.index_update(k, jo.index[t], k_)
    return spec, P_, p_, K, k

  _, P, p, K, k = fori_loop(
      0, T, rev_loop,
      (spec, spec.Q[T + 1], spec.q[T + 1], K, k))

  return K, k


def lqr_predict(spec, x0):
  """ Using dynamics from LQR spec and K,k from lqr_solve, calculates optimal trajectory (forward pass) """
  T, control_dim, _ = spec.R.shape
  _, state_dim, _ = spec.Q.shape

  K, k = lqr_solve(spec)

  def fwd_loop(t, state):
    spec, X, U = state
    A, B = spec.A[t], spec.B[t]
    u = mv(K[t], X[t]) + k[t]
    x = mv(A, X[t]) + mv(B, u)
    X = jo.index_update(X, jo.index[t + 1], x)
    U = jo.index_update(U, jo.index[t], u)
    return spec, X, U

  U = jnp.zeros((T, control_dim))
  X = jnp.zeros((T + 1, state_dim))
  X = jo.index_update(X, jo.index[0], x0)
  _, X, U = fori_loop(0, T, fwd_loop, (spec, X, U))
  return X, U


def ilqr(iterations, p, x0, U, X=None):
  """ Given nonlinear control spec (dynamics, cost, etc), iterations, initial state, and
  initial guess for input, repeatedly linearizes dynamics, cost and does LQR on the linearized
  model"""

  assert x0.ndim == 1 and x0.shape[0] == p.state_dim, x0.shape
  assert U.ndim > 0 and U.shape[0] == p.horizon, (U.shape, p.horizon)

  lqr_approx = make_lqr_approx(p)

  def loop(_, state):
    X, U = state
    p_lqr = lqr_approx(X, U)
    dX, dU = lqr_predict(p_lqr, jnp.zeros_like(x0))
    U = U + dU
    X = trajectory(p.dynamics, U, X[0] + dX[0])
    return X, U

  if (X is None):
    X = trajectory(p.dynamics, U, x0)
  return fori_loop(0, iterations, loop, (X, U))

#have to implement line search after this

def trans_cost(constraints, p):
  """ Returns a function that transforms cost function to include inequality constraints """
  """ The function returned should take in parameter t and return the cost function
  This function can't be that function because the new cost function is parameterized
  by the inequality constraints so we first write this function to create a 
  new function writer that already has the constraints implemented"""

  def t_fn(t):
    
    def constraint_cost(time, x, u):
      cost_sum = 0
      #use for i in 
      for constraint in constraints:
        cost_sum += (-1/t) * jnp.log(-constraint(x,u))
      return cost_sum

    def cost_fn(time, x, u):
      return jnp.reshape(p.cost(time, x, u) + constraint_cost(time, x, u), ()) 
    
    return cost_fn

  return t_fn

def cilqr(cilqr_iterations, ilqr_iterations, c_spec, x0, U, t):

  #ilqr
  """
  Things that don't change:
  Function that creates new cost function. That function is parameterized by the initial
  constraints. The only thing that really needs to change every iteration in that is 
  how much weight t has on the new costs. 
  Dynamics
  cilqr_iterations
  ilqr_iterations
  x0

  Things that change after each iteration:
    parameter t increases by scalar mu
    U, the initial trajectory input becomes the value returned by ilqr
    X, the optimal trajectory. Not sure if this trajectory should be from linearized
    dynamics or not. 
    Constraints are on the input and state. However, they are transformed by being added
    to the cost. So the dynamics of the system are the same 

  """
  assert t != 0

  mu = c_spec.mu
  constraints = c_spec.constraints
  p = c_spec.ControlSpec
  t_fn = trans_cost(constraints, p)

  def loop(_, state):
    X, U, t = state
    #transform inequality constraints and add them to the cost by creating new spec
    new_cost = t_fn(t)
    p_ = ControlSpec(new_cost, p.dynamics, p.horizon, p.state_dim, p.control_dim)
    X, U = ilqr(ilqr_iterations, p_, x0, U, X)
    t = mu * t
    print("X", X)
    print("U", U)
    return X, U, t

  #initial trajectory  
  X = trajectory(p.dynamics, U, x0)
  return fori_loop(0, cilqr_iterations, loop, (X, U, t))

def mpc_predict(solver, p, x0, U):
  assert x0.ndim == 1 and x0.shape[0] == p.state_dim
  T = p.horizon

  def zero_padded_controls_window(U, t):
    U_pad = jnp.vstack((U, jnp.zeros(U.shape)))
    return lax.dynamic_slice_in_dim(U_pad, t, T, axis=0)

  def loop(t, state):
    cost = lambda t_, x, u: p.cost(t + t_, x, u)
    dyns = lambda t_, x, u: p.dynamics(t + t_, x, u)

    X, U = state
    p_ = ControlSpec(cost, dyns, T, p.state_dim, p.control_dim)
    xt = X[t]
    U_rem = zero_padded_controls_window(U, t)
    _, U_ = solver(p_, xt, U_rem)
    ut = U_[0]
    x = p.dynamics(t, xt, ut)
    X = jo.index_update(X, jo.index[t + 1], x)
    U = jo.index_update(U, jo.index[t], ut)
    return X, U

  X = jnp.zeros((T + 1, p.state_dim))
  X = jo.index_update(X, jo.index[0], x0)
  return fori_loop(0, T, loop, (X, U))