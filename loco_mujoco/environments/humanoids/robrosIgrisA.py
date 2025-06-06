from typing import List, Union, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class RobrosIgrisA(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the ROBROS IGRIS-A robot.


    Default Observation Space
    -----------------

    ============ ================== ================ ==================================== ============================== ===
    Index in Obs Name               ObservationType  Min                                  Max                            Dim
    ============ ================== ================ ==================================== ============================== ===
    0 - 4        q_root             FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    5            q_back_bkz         JointPos         [-2.35]                              [2.35]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    6            q_l_arm_shy        JointPos         [-2.87]                              [2.87]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    7            q_l_arm_shx        JointPos         [-0.34]                              [3.11]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    8            q_l_arm_shz        JointPos         [-1.3]                               [4.45]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    9            q_left_elbow       JointPos         [-1.25]                              [2.61]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    10           q_r_arm_shy        JointPos         [-2.87]                              [2.87]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    11           q_r_arm_shx        JointPos         [-3.11]                              [0.34]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    12           q_r_arm_shz        JointPos         [-4.45]                              [1.3]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    13           q_right_elbow      JointPos         [-1.25]                              [2.61]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    14           q_hip_flexion_r    JointPos         [-1.57]                              [1.57]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    15           q_hip_adduction_r  JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    16           q_hip_rotation_r   JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    17           q_knee_angle_r     JointPos         [-0.26]                              [2.05]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    18           q_ankle_angle_r    JointPos         [-0.87]                              [0.52]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    19           q_hip_flexion_l    JointPos         [-1.57]                              [1.57]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    20           q_hip_adduction_l  JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    21           q_hip_rotation_l   JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    22           q_knee_angle_l     JointPos         [-0.26]                              [2.05]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    23           q_ankle_angle_l    JointPos         [-0.87]                              [0.52]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    24 - 29      dq_root            FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    30           dq_back_bkz        JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    31           dq_l_arm_shy       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    32           dq_l_arm_shx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    33           dq_l_arm_shz       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    34           dq_left_elbow      JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    35           dq_r_arm_shy       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    36           dq_r_arm_shx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    37           dq_r_arm_shz       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    38           dq_right_elbow     JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    39           dq_hip_flexion_r   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    40           dq_hip_adduction_r JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    41           dq_hip_rotation_r  JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    42           dq_knee_angle_r    JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    43           dq_ankle_angle_r   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    44           dq_hip_flexion_l   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    45           dq_hip_adduction_l JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    46           dq_hip_rotation_l  JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    47           dq_knee_angle_l    JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    48           dq_ankle_angle_l   JointVel         [-inf]                               [inf]                          1
    ============ ================== ================ ==================================== ============================== ===

    Default Action Space
    -----------------

    Control function type: **DefaultControl**

    See control function interface for more details.

    =============== ==== ===
    Index in Action Min  Max
    =============== ==== ===
    0               -1.0 1.0
    --------------- ---- ---
    1               -1.0 1.0
    --------------- ---- ---
    2               -1.0 1.0
    --------------- ---- ---
    3               -1.0 1.0
    --------------- ---- ---
    4               -1.0 1.0
    --------------- ---- ---
    5               -1.0 1.0
    --------------- ---- ---
    6               -1.0 1.0
    --------------- ---- ---
    7               -1.0 1.0
    --------------- ---- ---
    8               -1.0 1.0
    --------------- ---- ---
    9               -1.0 1.0
    --------------- ---- ---
    10              -1.0 1.0
    --------------- ---- ---
    11              -1.0 1.0
    --------------- ---- ---
    12              -1.0 1.0
    --------------- ---- ---
    13              -1.0 1.0
    --------------- ---- ---
    14              -1.0 1.0
    --------------- ---- ---
    15              -1.0 1.0
    --------------- ---- ---
    16              -1.0 1.0
    --------------- ---- ---
    17              -1.0 1.0
    --------------- ---- ---
    18              -1.0 1.0
    =============== ==== ===


    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(self, disable_arms: bool = False, disable_back_joint: bool = False,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_arms (bool): Whether to disable arm joints.
            disable_back_joint (bool): Whether to disable the back joint.
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """

        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)
        if disable_arms or disable_back_joint:
            joints_to_remove, actuators_to_remove, equ_constraints_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in actuators_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          actuators_to_remove, equ_constraints_to_remove)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_spec_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, actuators, and equality constraints should be removed from the Mujoco specification.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, actuators to remove,
            and equality constraints to remove.
        """

        joints_to_remove = []
        actuators_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["Joint_1_1_1_Shoulder_Pitch_Left_Arm", "Joint_1_1_2_Shoulder_Roll_Left_Arm", "Joint_1_1_3_Shoulder_Yaw_Left_Arm", "Joint_1_1_4_Elbow_Left_Arm", "Joint_1_1_5_Wrist_Yaw_Left_Arm", "Joint_1_1_6_Wrist_Pitch_Left_Arm", "Joint_1_1_7_Wrist_Roll_Left_Arm", 
                                 "Joint_1_2_1_Shoulder_Pitch_Right_Arm", "Joint_1_2_2_Shoulder_Roll_Right_Arm", "Joint_1_2_3_Shoulder_Yaw_Right_Arm", "Joint_1_2_4_Elbow_Right_Arm", "Joint_1_2_5_Wrist_Yaw_Right_Arm", "Joint_1_2_6_Wrist_Pitch_Right_Arm", "Joint_1_2_7_Wrist_Roll_Right_Arm"]
            actuators_to_remove += ["Joint_1_1_1_Shoulder_Pitch_Left_Arm", "Joint_1_1_2_Shoulder_Roll_Left_Arm", "Joint_1_1_3_Shoulder_Yaw_Left_Arm", "Joint_1_1_4_Elbow_Left_Arm", "Joint_1_1_5_Wrist_Yaw_Left_Arm", "Joint_1_1_6_Wrist_Pitch_Left_Arm", "Joint_1_1_7_Wrist_Roll_Left_Arm", 
                                 "Joint_1_2_1_Shoulder_Pitch_Right_Arm", "Joint_1_2_2_Shoulder_Roll_Right_Arm", "Joint_1_2_3_Shoulder_Yaw_Right_Arm", "Joint_1_2_4_Elbow_Right_Arm", "Joint_1_2_5_Wrist_Yaw_Right_Arm", "Joint_1_2_6_Wrist_Pitch_Right_Arm", "Joint_1_2_7_Wrist_Roll_Right_Arm"]

        if self._disable_back_joint:
            joints_to_remove += ["Joint_3_1_Waist_Yaw", "Joint_3_2_Waist_Pitch", "Joint_3_3_Waist_Roll"]
            actuators_to_remove += ["Joint_3_1_Waist_Yaw", "Joint_3_2_Waist_Pitch", "Joint_3_3_Waist_Roll"]

        return joints_to_remove, actuators_to_remove, equ_constr_to_remove


    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: List of observations.
        """
        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            ObservationType.JointPos("q_Joint_2_1_1_Hip_Roll_Left_Leg", xml_name="Joint_2_1_1_Hip_Roll_Left_Leg"),
                            ObservationType.JointPos("q_Joint_2_1_2_Hip_Yaw_Left_Leg", xml_name="Joint_2_1_2_Hip_Yaw_Left_Leg"),
                            ObservationType.JointPos("q_Joint_2_1_3_Hip_Pitch_Left_Leg", xml_name="Joint_2_1_3_Hip_Pitch_Left_Leg"),
                            ObservationType.JointPos("q_Joint_2_1_4_Knee_Left_Leg", xml_name="Joint_2_1_4_Knee_Left_Leg"),
                            ObservationType.JointPos("q_Joint_2_1_5_Ankle_Pitch_Left_Leg", xml_name="Joint_2_1_5_Ankle_Pitch_Left_Leg"),
                            ObservationType.JointPos("q_Joint_2_1_6_Ankle_Roll_Left_Leg", xml_name="Joint_2_1_6_Ankle_Roll_Left_Leg"),
                            
                            ObservationType.JointPos("q_Joint_2_2_1_Hip_Roll_Right_Leg", xml_name="Joint_2_2_1_Hip_Roll_Right_Leg"),
                            ObservationType.JointPos("q_Joint_2_2_2_Hip_Yaw_Right_Leg", xml_name="Joint_2_2_2_Hip_Yaw_Right_Leg"),
                            ObservationType.JointPos("q_Joint_2_2_3_Hip_Pitch_Right_Leg", xml_name="Joint_2_2_3_Hip_Pitch_Right_Leg"),
                            ObservationType.JointPos("q_Joint_2_2_4_Knee_Right_Leg", xml_name="Joint_2_2_4_Knee_Right_Leg"),
                            ObservationType.JointPos("q_Joint_2_2_5_Ankle_Pitch_Right_Leg", xml_name="Joint_2_2_5_Ankle_Pitch_Right_Leg"),
                            ObservationType.JointPos("q_Joint_2_2_6_Ankle_Roll_Right_Leg", xml_name="Joint_2_2_6_Ankle_Roll_Right_Leg"),
                            
                            ObservationType.JointPos("q_Joint_3_1_Waist_Yaw", xml_name="Joint_3_1_Waist_Yaw"),
                            ObservationType.JointPos("q_Joint_3_2_Waist_Pitch", xml_name="Joint_3_2_Waist_Pitch"),
                            ObservationType.JointPos("q_Joint_3_3_Waist_Roll", xml_name="Joint_3_3_Waist_Roll"),
                            
                            ObservationType.JointPos("q_Joint_1_1_1_Shoulder_Pitch_Left_Arm", xml_name="Joint_1_1_1_Shoulder_Pitch_Left_Arm"),
                            ObservationType.JointPos("q_Joint_1_1_2_Shoulder_Roll_Left_Arm", xml_name="Joint_1_1_2_Shoulder_Roll_Left_Arm"),
                            ObservationType.JointPos("q_Joint_1_1_3_Shoulder_Yaw_Left_Arm", xml_name="Joint_1_1_3_Shoulder_Yaw_Left_Arm"),
                            ObservationType.JointPos("q_Joint_1_1_4_Elbow_Left_Arm", xml_name="Joint_1_1_4_Elbow_Left_Arm"),
                            ObservationType.JointPos("q_Joint_1_1_5_Wrist_Yaw_Left_Arm", xml_name="Joint_1_1_5_Wrist_Yaw_Left_Arm"),
                            ObservationType.JointPos("q_Joint_1_1_6_Wrist_Pitch_Left_Arm", xml_name="Joint_1_1_6_Wrist_Pitch_Left_Arm"),
                            ObservationType.JointPos("q_Joint_1_1_7_Wrist_Roll_Left_Arm", xml_name="Joint_1_1_7_Wrist_Roll_Left_Arm"),
                            
                            ObservationType.JointPos("q_Joint_1_2_1_Shoulder_Pitch_Right_Arm", xml_name="Joint_1_2_1_Shoulder_Pitch_Right_Arm"),
                            ObservationType.JointPos("q_Joint_1_2_2_Shoulder_Roll_Right_Arm", xml_name="Joint_1_2_2_Shoulder_Roll_Right_Arm"),
                            ObservationType.JointPos("q_Joint_1_2_3_Shoulder_Yaw_Right_Arm", xml_name="Joint_1_2_3_Shoulder_Yaw_Right_Arm"),
                            ObservationType.JointPos("q_Joint_1_2_4_Elbow_Right_Arm", xml_name="Joint_1_2_4_Elbow_Right_Arm"),
                            ObservationType.JointPos("q_Joint_1_2_5_Wrist_Yaw_Right_Arm", xml_name="Joint_1_2_5_Wrist_Yaw_Right_Arm"),
                            ObservationType.JointPos("q_Joint_1_2_6_Wrist_Pitch_Right_Arm", xml_name="Joint_1_2_6_Wrist_Pitch_Right_Arm"),
                            ObservationType.JointPos("q_Joint_1_2_7_Wrist_Roll_Right_Arm", xml_name="Joint_1_2_7_Wrist_Roll_Right_Arm"),
                            

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_Joint_2_1_1_Hip_Roll_Left_Leg", xml_name="Joint_2_1_1_Hip_Roll_Left_Leg"),
                            ObservationType.JointVel("dq_Joint_2_1_2_Hip_Yaw_Left_Leg", xml_name="Joint_2_1_2_Hip_Yaw_Left_Leg"),
                            ObservationType.JointVel("dq_Joint_2_1_3_Hip_Pitch_Left_Leg", xml_name="Joint_2_1_3_Hip_Pitch_Left_Leg"),
                            ObservationType.JointVel("dq_Joint_2_1_4_Knee_Left_Leg", xml_name="Joint_2_1_4_Knee_Left_Leg"),
                            ObservationType.JointVel("dq_Joint_2_1_5_Ankle_Pitch_Left_Leg", xml_name="Joint_2_1_5_Ankle_Pitch_Left_Leg"),
                            ObservationType.JointVel("dq_Joint_2_1_6_Ankle_Roll_Left_Leg", xml_name="Joint_2_1_6_Ankle_Roll_Left_Leg"),
                            
                            ObservationType.JointVel("dq_Joint_2_2_1_Hip_Roll_Right_Leg", xml_name="Joint_2_2_1_Hip_Roll_Right_Leg"),
                            ObservationType.JointVel("dq_Joint_2_2_2_Hip_Yaw_Right_Leg", xml_name="Joint_2_2_2_Hip_Yaw_Right_Leg"),
                            ObservationType.JointVel("dq_Joint_2_2_3_Hip_Pitch_Right_Leg", xml_name="Joint_2_2_3_Hip_Pitch_Right_Leg"),
                            ObservationType.JointVel("dq_Joint_2_2_4_Knee_Right_Leg", xml_name="Joint_2_2_4_Knee_Right_Leg"),
                            ObservationType.JointVel("dq_Joint_2_2_5_Ankle_Pitch_Right_Leg", xml_name="Joint_2_2_5_Ankle_Pitch_Right_Leg"),
                            ObservationType.JointVel("dq_Joint_2_2_6_Ankle_Roll_Right_Leg", xml_name="Joint_2_2_6_Ankle_Roll_Right_Leg"),
                            
                            ObservationType.JointVel("dq_Joint_3_1_Waist_Yaw", xml_name="Joint_3_1_Waist_Yaw"),
                            ObservationType.JointVel("dq_Joint_3_2_Waist_Pitch", xml_name="Joint_3_2_Waist_Pitch"),
                            ObservationType.JointVel("dq_Joint_3_3_Waist_Roll", xml_name="Joint_3_3_Waist_Roll"),
                            
                            ObservationType.JointVel("dq_Joint_1_1_1_Shoulder_Pitch_Left_Arm", xml_name="Joint_1_1_1_Shoulder_Pitch_Left_Arm"),
                            ObservationType.JointVel("dq_Joint_1_1_2_Shoulder_Roll_Left_Arm", xml_name="Joint_1_1_2_Shoulder_Roll_Left_Arm"),
                            ObservationType.JointVel("dq_Joint_1_1_3_Shoulder_Yaw_Left_Arm", xml_name="Joint_1_1_3_Shoulder_Yaw_Left_Arm"),
                            ObservationType.JointVel("dq_Joint_1_1_4_Elbow_Left_Arm", xml_name="Joint_1_1_4_Elbow_Left_Arm"),
                            ObservationType.JointVel("dq_Joint_1_1_5_Wrist_Yaw_Left_Arm", xml_name="Joint_1_1_5_Wrist_Yaw_Left_Arm"),
                            ObservationType.JointVel("dq_Joint_1_1_6_Wrist_Pitch_Left_Arm", xml_name="Joint_1_1_6_Wrist_Pitch_Left_Arm"),
                            ObservationType.JointVel("dq_Joint_1_1_7_Wrist_Roll_Left_Arm", xml_name="Joint_1_1_7_Wrist_Roll_Left_Arm"),
                            
                            ObservationType.JointVel("dq_Joint_1_2_1_Shoulder_Pitch_Right_Arm", xml_name="Joint_1_2_1_Shoulder_Pitch_Right_Arm"),
                            ObservationType.JointVel("dq_Joint_1_2_2_Shoulder_Roll_Right_Arm", xml_name="Joint_1_2_2_Shoulder_Roll_Right_Arm"),
                            ObservationType.JointVel("dq_Joint_1_2_3_Shoulder_Yaw_Right_Arm", xml_name="Joint_1_2_3_Shoulder_Yaw_Right_Arm"),
                            ObservationType.JointVel("dq_Joint_1_2_4_Elbow_Right_Arm", xml_name="Joint_1_2_4_Elbow_Right_Arm"),
                            ObservationType.JointVel("dq_Joint_1_2_5_Wrist_Yaw_Right_Arm", xml_name="Joint_1_2_5_Wrist_Yaw_Right_Arm"),
                            ObservationType.JointVel("dq_Joint_1_2_6_Wrist_Pitch_Right_Arm", xml_name="Joint_1_2_6_Wrist_Pitch_Right_Arm"),
                            ObservationType.JointVel("dq_Joint_1_2_7_Wrist_Roll_Right_Arm", xml_name="Joint_1_2_7_Wrist_Roll_Right_Arm"),
                            ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of action names.
        """
        action_spec = ["Joint_2_1_1_Hip_Roll_Left_Leg", "Joint_2_1_2_Hip_Yaw_Left_Leg", "Joint_2_1_3_Hip_Pitch_Left_Leg", "Joint_2_1_4_Knee_Left_Leg", "Joint_2_1_5_Ankle_Pitch_Left_Leg", "Joint_2_1_6_Ankle_Roll_Left_Leg", 
                       "Joint_2_2_1_Hip_Roll_Right_Leg", "Joint_2_2_2_Hip_Yaw_Right_Leg", "Joint_2_2_3_Hip_Pitch_Right_Leg", "Joint_2_2_4_Knee_Right_Leg", "Joint_2_2_5_Ankle_Pitch_Right_Leg", "Joint_2_2_6_Ankle_Roll_Right_Leg", 
                       "Joint_3_1_Waist_Yaw", "Joint_3_2_Waist_Pitch", "Joint_3_3_Waist_Roll",
                       "Joint_1_1_1_Shoulder_Pitch_Left_Arm", "Joint_1_1_2_Shoulder_Roll_Left_Arm", "Joint_1_1_3_Shoulder_Yaw_Left_Arm", "Joint_1_1_4_Elbow_Left_Arm", "Joint_1_1_5_Wrist_Yaw_Left_Arm", "Joint_1_1_6_Wrist_Pitch_Left_Arm", "Joint_1_1_7_Wrist_Roll_Left_Arm",
                       "Joint_1_2_1_Shoulder_Pitch_Right_Arm", "Joint_1_2_2_Shoulder_Roll_Right_Arm", "Joint_1_2_3_Shoulder_Yaw_Right_Arm", "Joint_1_2_4_Elbow_Right_Arm", "Joint_1_2_5_Wrist_Yaw_Right_Arm", "Joint_1_2_6_Wrist_Pitch_Right_Arm", "Joint_1_2_7_Wrist_Roll_Right_Arm",
                       ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the Unitree H1 environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "robros_igris" / "igris_v1_2.xml").as_posix()

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body specified in the XML file.
        """
        return "3_3_Torso"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root specified in the XML file.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.5, 1.5)
