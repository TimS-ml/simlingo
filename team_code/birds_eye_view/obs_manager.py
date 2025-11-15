"""Observation Manager Base Class - Interface for BEV Renderers.

This module defines the base class interface for observation managers that
generate bird's-eye-view (BEV) representations of the driving scene.

Observation managers are responsible for:
1. Defining the observation space (dimensions, channels, dtypes)
2. Attaching to an ego vehicle to track its state
3. Rendering BEV observations at each timestep

Different observation managers can implement different rendering styles
(e.g., ChauffeurNet semantic maps, occupancy grids, etc.).

Code adapted from: https://github.com/zhejz/carla-roach

Classes:
    ObsManagerBase: Abstract base class for BEV observation generators
"""


class ObsManagerBase(object):
  """
  base class for observation managers
  """

  def __init__(self):
    self._define_obs_space()

  def _define_obs_space(self):
    raise NotImplementedError

  def attach_ego_vehicle(self, parent_actor):
    raise NotImplementedError

  def get_observation(self):
    raise NotImplementedError

  def clean(self):
    raise NotImplementedError
