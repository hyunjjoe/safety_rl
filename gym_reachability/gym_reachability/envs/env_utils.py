"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
import numpy as np


# == margin ==
def calculate_margin_rect(s, x_y_w_h, negativeInside=True):
  """Calculates the margin to a rectangular box in the x-y state space.

    Args:
        s (np.ndarray): the state of the agent. It requires that s[0] is the
            x position and s[1] is the y position.
        x_y_w_h (tuple of floats): (center_x, center_y, width, height).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the box. Defaults to True.

    Returns:
        float: margin.
    """
  x, y, w, h = x_y_w_h
  delta_x = np.abs(s[0] - x)
  delta_y = np.abs(s[1] - y)
  margin = max(delta_y - h/2, delta_x - w/2)

  if negativeInside:
    return margin
  else:
    return -margin

def calculate_signed_distance(eef_pos, object_pos, object_dims):
    """Calculates the signed distance from the end effector to the object.

    Args:
        eef_pos (np.ndarray): The position of the end effector [x, y, z].
        object_pos (np.ndarray): The position of the object center [x, y, z].
        object_dims (tuple): The dimensions of the object (width, height, depth).

    Returns:
        float: The signed distance. Negative if the end effector is inside the object.
    """
    # Calculate the Euclidean distance
    distance = np.linalg.norm(eef_pos - object_pos)

    # Check if the end effector is inside the object
    inside = np.all(np.abs(eef_pos - object_pos) <= np.array(object_dims))

    # Return the signed distance
    return -distance if inside else distance

def calculate_margin_cube(s, center_xyz_whd, negativeInside=True):
    """Calculates the margin to a cubic box in the x-y-z state space.

    Args:
        s (np.ndarray): the state of the agent. It requires that s[0] is the
            x position, s[1] is the y position, and s[2] is the z position.
        center_xyz_whd (tuple of floats): (center_x, center_y, center_z, width, height, depth).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the box. Defaults to True.

    Returns:
        float: margin.
    """
    x, y, z, w, h, d = center_xyz_whd
    delta_x = np.abs(s[0] - x)
    delta_y = np.abs(s[1] - y)
    delta_z = np.abs(s[2] - z)
    margin = max(delta_x - w / 2, delta_y - h / 2, delta_z - d / 2)

    if negativeInside:
        return margin
    else:
        return -margin

def calculate_margin_circle(s, c_r, negativeInside=True):
  """Calculates the margin to a circle in the x-y state space.

    Args:
        s (np.ndarray): the state of the agent. It requires that s[0] is the
            x position and s[1] is the y position.
        c_r (tuple of np.ndarray and float)): (center, radius).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the circle. Defaults to True.

    Returns:
        float: margin.
    """
  center, radius = c_r
  dist_to_center = np.linalg.norm(s[:2] - center)
  margin = dist_to_center - radius

  if negativeInside:
    return margin
  else:
    return -margin

def calculate_margin_circle_param(s, c_r, negativeInside=True):
  x_center, radius = c_r
  dist_to_center = np.linalg.norm(s[:2] - np.array([x_center, s[-1]]))
  margin = dist_to_center - radius
  if negativeInside:
    return margin
  else:
    return -margin

# == Plotting ==
def plot_arc(
    center, r, thetaParam, ax, c='b', lw=1.5, orientation=0., zorder=0
):
  """Plots an arc given a center, a radius and the (theta_init, theta_final).

  Args:
      center (np.ndarray): the center of the arc.
      r (float): the radius of the arc.
      thetaParam (np.ndarray): the initial angle and the final angle.
      ax (matplotlib.axes.Axes): ax to plot.
      c (str, optional): color of the arc. Defaults to 'b'.
      lw (float, optional): linewidth of the arc. Defaults to 1.5.
      orientation (float, optional): rotate points in the xy-plane
          counterclockwise through orientation with respect to the x axis.
          Defaults to 0.
      zorder (int, optional): graph layers order. Defaults to 0.
  """
  x, y = center
  thetaInit, thetaFinal = thetaParam

  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(thetaInit + orientation, thetaFinal + orientation, 100)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)

  ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)


def plot_circle(
    center, r, ax, c='b', lw=1.5, ls='-', orientation=0, scatter=False,
    zorder=0
):
  """Plots a circle given a center and a radius.

  Args:
      enter (np.ndarray): the center of the arc.
      r (float): the radius of the arc.
      ax (matplotlib.axes.Axes): ax to plot.
      c (str, optional): color of the circle. Defaults to 'b'.
      lw (float, optional): linewidth of the circle. Defaults to 1.5.
      ls (str, optional): linestyle of the circle. Defaults to '-'.
      orientation (int, optional): rotate points in the xy-plane
          counterclockwise through orientation with respect to the x axis.
          Defaults to 0.
      scatter (bool, optional): show the centerif True. Defaults to False.
      zorder (int, optional): graph layers order. Defaults to 0.
  """
  x, y = center
  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(0, 2 * np.pi, 200)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)
  ax.plot(xs, ys, c=c, lw=lw, linestyle=ls, zorder=zorder)
  if scatter:
    ax.scatter(xtilde + r, ytilde, c=c, s=80)
    ax.scatter(xtilde - r, ytilde, c=c, s=80)
    print(xtilde + r, ytilde, xtilde - r, ytilde)


def rotatePoint(state, orientation):
  """Rotates the point counter-clockwise by a given angle.

  Args:
      state (np.ndarray): the state of the agent. It requires that s[0] is the
          x position and s[1] is the y position.
      orientation (int, optional): counter-clockwise angle.

  Returns:
      np.ndarray: rotated state.
  """
  x, y, theta = state
  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)
  thetatilde = theta + orientation

  return np.array([xtilde, ytilde, thetatilde])
