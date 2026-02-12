import os
import rclpy
import threading
# import mujoco_py
import mujoco as mj
import mujoco.viewer 
import math
import os
import time
from dynamixel_sdk import *
from custom_interfaces.srv import GetPositions, SetPositions
# from custom_interfaces.msg import SetPositions
import rclpy.logging
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

class MinimalService(Node) :
    global packetHandler, portHandler, DXL_IDs, positions_deg
   
    def __init__(self):
        super().__init__('dclaw_sim_py_node')
        self.declare_parameter(
            'mujoco_model_path',
            '/home/songwoo/Desktop/work_dir/realsense_hand_retargetting/universal_robots_ur5e_with_dclaw/dclaw_two_finger/dclaw3xh.xml',
        )
        self.declare_parameter('stiffness_profile_csv', '')
        self.declare_parameter('stiffness_columns', 'Kx,Ky,Kz')
        self.declare_parameter('stiffness_time_column', 'time')
        self.declare_parameter('reference_profile_csv', '')
        self.declare_parameter('reference_columns', 'thumb_x,thumb_y,thumb_z')
        self.declare_parameter('reference_time_column', 'time')

        model_path_param = str(self.get_parameter('mujoco_model_path').value or '')
        if model_path_param:
            model_path = Path(model_path_param).expanduser()
        else:
            model_path = Path()
        if not model_path_param or not model_path.exists():
            pkg_path = os.path.dirname(os.path.abspath(__name__))
            fallback = Path(pkg_path) / 'src/dclaw/resource/robel_sim/dclaw/dclaw3xh.xml'
            self.get_logger().warning(
                f"Specified Mujoco model path '{model_path}' not found. Falling back to '{fallback}'."
            )
            model_path = fallback

        self.model = mj.MjModel.from_xml_path(str(model_path))    
        self.data = mj.MjData(self.model)
        self.positions_deg = [.0, .0, .0, .0, .0, .0, .0, .0, .0] # 10 11 12 20 21 22 30 31 32
        self.torques = np.array([.0, .0, .0, .0, .0, .0, .0, .0, .0],dtype=float) # 10 11 12 20 21 22 30 31 32
        self.timestamp = 0
        self.sensordata = np.zeros([3,6])
        self.dt = 0.001
        self.model.opt.timestep = self.dt  # 더 긴 시뮬레이션 시간 간격을 위해 timestep을 늘림

        self.default_goals = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1]], dtype=float)
        self.default_stiffness = np.eye(3) * 200.0
        self.default_damping = np.eye(3) * 1.0
        self.default_inertia = np.eye(3) * 0.01
        self.sensor_constant = 20.0
        self.kp = 250.0
        self.kd = 1.0
        self.pd_flag = True

        stiffness_cols_param = str(self.get_parameter('stiffness_columns').value or '')
        reference_cols_param = str(self.get_parameter('reference_columns').value or '')
        stiffness_time_col = str(self.get_parameter('stiffness_time_column').value or '')
        reference_time_col = str(self.get_parameter('reference_time_column').value or '')

        stiffness_cols = [c.strip() for c in stiffness_cols_param.split(',') if c.strip()]
        reference_cols = [c.strip() for c in reference_cols_param.split(',') if c.strip()]

        stiffness_csv = str(self.get_parameter('stiffness_profile_csv').value or '')
        reference_csv = str(self.get_parameter('reference_profile_csv').value or '')

        self.stiffness_times, self.stiffness_profile = self._load_profile(
            stiffness_csv, stiffness_cols, stiffness_time_col
        )
        self.reference_times, self.reference_profile = self._load_profile(
            reference_csv, reference_cols, reference_time_col
        )

        rclpy.logging.get_logger('dclaw_sim_py_node').info('\n\n HIHI \n\n')
        
        # 1) get/set robot state from real node
        self.srv = self.create_service(SetPositions, 'set_positions', self.get_present_pos)

        # 2) simulate robot and calcuate robot state
        self.thread = threading.Thread(target=self.simulate, daemon=True)  
        self.thread.start()


    def _load_profile(
        self,
        csv_path: str,
        columns: List[str],
        time_column: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not csv_path:
            return None, None
        path = Path(str(csv_path)).expanduser()
        if not path.exists():
            self.get_logger().warning(f"Profile CSV not found: {path}")
            return None, None
        try:
            data = np.genfromtxt(str(path), delimiter=',', names=True, dtype=float)
        except Exception as exc:
            self.get_logger().error(f"Failed to load profile from {path}: {exc}")
            return None, None

        if data.dtype.names is None:
            self.get_logger().warning(f"Profile CSV {path} is missing a header row; skipping.")
            return None, None

        if data.shape == ():
            data = np.array([data])

        missing = [col for col in columns if col not in data.dtype.names]
        if missing:
            self.get_logger().warning(
                f"Profile CSV {path} missing expected columns: {missing}; available={list(data.dtype.names)}"
            )
            return None, None

        values = np.vstack([np.atleast_1d(data[col]) for col in columns]).T.astype(float)
        if values.ndim == 1:
            values = values.reshape(-1, len(columns))

        if time_column and time_column in data.dtype.names:
            times = np.atleast_1d(data[time_column]).astype(float)
        else:
            times = np.arange(values.shape[0], dtype=float) * self.dt

        self.get_logger().info(
            f"Loaded profile '{path.name}' with {values.shape[0]} samples and columns={columns}"
        )
        return times, values


    @staticmethod
    def _sample_profile(
        times: Optional[np.ndarray],
        values: Optional[np.ndarray],
        t: float,
    ) -> Optional[np.ndarray]:
        if times is None or values is None or values.size == 0:
            return None
        if t <= times[0]:
            return values[0].astype(float)
        if t >= times[-1]:
            return values[-1].astype(float)
        return np.array([np.interp(t, times, values[:, idx]) for idx in range(values.shape[1])], dtype=float)

    
    def get_present_pos(self, req, res):
        
        ids = req.ids
        # -------------------------------------------
        pos_deg = req.positions
        self.sensordata = np.array(list(req.sensordata)).reshape([3,6])
        # -------------------------------------------
        temp_sensor = self.sensordata[0:3,0:3].copy()
        self.sensordata[0:3,0] = temp_sensor[0:3,1]
        self.sensordata[0:3,1] = temp_sensor[0:3,0]
        # self.sensordata = self.sensordata[0:3][1]

        for idx, id in enumerate(ids) :
            if id in [11,12,21,22,31,32] :
                self.positions_deg[idx] = pos_deg[idx] - 3.14
            else :
                self.positions_deg[idx] = pos_deg[idx] - 1.57

        res.torques = np.asarray(self.torques, dtype=float).tolist()
        res.timestamp = self.timestamp
        print(f'response : {self.timestamp}')
        return res

    def simulate(self):
  
        goals = [[0., .0, 0.1],[0., .0, 0.1],[0., .0, 0.1]] #Desire position 2
        
        damping = 0.25
        #Init parameters
        jacp = np.zeros((3,3, self.model.nv)) #translation jacobian (NUMBER OF JOINT x NUM_OF_ACTUATORS)
        jacr = np.zeros((3,3, self.model.nv)) #rotational jacobian
        #-------------------------------------------Parameter ---------------------------------------------
        # desired_stiffness = [[300, 0, 0],[0, 300, 0],[0, 0, 300]]
        desired_stiffness = np.eye(3) * 200
        desired_damping = np.eye(3) * 1
        desired_inertia = np.eye(3) * 0.01 # compare 1 with 30
        # sensor_constant = 5 # paper cup
        sensor_constant = 20
        kp = 250
        kd = 1
        pd_flag = True 

        # -------------------------------------------------------------------------------------------------
        #Get error.
        end_effector_id = []
        end_effector_id.append(self.model.body('FFL12').id)
        end_effector_id.append(self.model.body('MFL22').id)
        end_effector_id.append(self.model.body('THL32').id)

        # current_pose = data.body(end_effector_id).xpos #Current pose
        # x_error = np.subtract(goal, current_pose) #Init Error

        
        forces_list = []
        ee_list = []
        time = 0
        tips = ["FFtip", "MFtip", "THtip"]    # 
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            time += 1
            with viewer.lock():
                for goal in goals:
                    viewer.user_scn.ngeom += 1

                    # initial setting : geom, type, size, pos, rot, rgba
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                            mujoco.mjtGeom.mjGEOM_CAPSULE, 
                            np.zeros(3), np.zeros(3), np.zeros(9), np.array([1, 0., 0., 1]))

                    # change setting : change value of geom
                    mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                            mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01,
                            np.array(goal)-0.001,
                            np.array(goal))
                
            self.timestamp = 0
            Jacp_1 = jacp.copy()
            # Jacp_2 = np.zeros([3,3])
            while viewer.is_running() and rclpy.ok():
                
                self.timestamp += 1
                tau_imp = 0
                tau_pd = 0
                for idx, tip in enumerate(tips):
                    x_error = np.subtract(goals[idx], self.data.site(tip).xpos)

                    rotation_matrix = self.data.site(tip).xmat.reshape(3, 3)
                    # start_point = self.data.site(tip).xpos  # 벡터 시작점: end-effector의 위치                    
                    # end_point_z = rotation_matrix @ np.array([0,0,1]) / np.linalg.norm(rotation_matrix @ np.array([0,0,1]))
                    # end_point_y = rotation_matrix @np.array([0,1,0]) / np.linalg.norm(rotation_matrix @ np.array([0,1,0]))
                    # end_point = rotation_matrix @ np.array([1,0,0]) / np.linalg.norm(rotation_matrix @ np.array([1,0,0]))

                    # 결과 출력
                    # print("Global Force Vector:", f_ext)
                    
                    mujoco.mj_jac(self.model, self.data, jacp[idx, :], jacr[idx, :], self.data.site(tip).xpos, end_effector_id[idx])
                    n = jacp[idx, :].shape[1]
                    I = np.identity(n)
                    product = jacp[idx, :].T @ jacp[idx, :] + damping * I

                    if np.isclose(np.linalg.det(product), 0):
                        j_inv = np.linalg.pinv(product) @ jacp[idx, :].T
                    else:
                        j_inv = np.linalg.inv(product) @ jacp[idx, :].T
                        

                    xvel = jacp[idx, :]@self.data.qvel 
                # -------------------------------------------
                    Jacp_2 = jacp.copy()
                    Jacp_dot = (Jacp_2[idx, :] - Jacp_1[idx, :]) / self.dt
                    Jacp_1 = Jacp_2.copy()
                # -------------------------------------------

                    M = np.zeros((self.model.nv, self.model.nv)) # 9x9
                    mj.mj_fullM(self.model, M, self.data.qM)

                    c = self.data.qfrc_bias # bais force (centrifugal, coriolis, gravity)
                    # c[3*len(tips):] = 0

                    Md_inv = np.linalg.inv(desired_inertia) 
                    
                    f_ext = np.multiply(rotation_matrix @ self.sensordata[idx][0:3],sensor_constant)#data.sensor("ft_sensor_force").self.data.copy() * 0
                    
                    # print(f"\n\n\{(jacp.T).shape}\n\n\n")    
                    tau_pd += jacp[idx, :].T@(kp*x_error + kd*xvel)
                    tau_imp += M@j_inv@(Md_inv@(-f_ext-desired_damping@xvel-desired_stiffness@x_error)-Jacp_dot@self.data.qvel)+c

                # print(f"\n\n\{np.round(jacp[1,0:3].T@f_ext,4)}\n\n\n")
                # print(f"\n\n\{tau_pd.shape}\n\n\n")
                if pd_flag == True:
                    self.torques = tau_pd
                else :
                    self.torques = -tau_imp
                # with viewer.lock():
                #     viewer.user_scn.ngeom += 1

                #     mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                #             mujoco.mjtGeom.mjGEOM_CAPSULE, 
                #             np.zeros(3), 
                #             np.zeros(3), 
                #             np.zeros(9), 
                #             np.array([1, 0., 0., 1]))

                #     # change setting : change value of geom
                #     mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                #             mujoco.mjtGeom.mjGEOM_CAPSULE, 
                #             0.005,
                #             start_point,
                #             end_point)    
                #     viewer.user_scn.ngeom += 1

                #     mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                #             mujoco.mjtGeom.mjGEOM_CAPSULE, 
                #             np.zeros(3), 
                #             np.zeros(3), 
                #             np.zeros(9), 
                #             np.array([0, 1., 0., 1])) # green

                #     # change setting : change value of geom
                #     mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                #             mujoco.mjtGeom.mjGEOM_CAPSULE, 
                #             0.005,
                #             start_point,
                #             end_point_y)    
                #     viewer.user_scn.ngeom += 1

                #     mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                #             mujoco.mjtGeom.mjGEOM_CAPSULE, 
                #             np.zeros(3), 
                #             np.zeros(3), 
                #             np.zeros(9), 
                #             np.array([0, 0., 1., 1])) # blue

                #     # change setting : change value of geom
                #     mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                #             mujoco.mjtGeom.mjGEOM_CAPSULE, 
                #             0.005,
                #             start_point,
                #             end_point_z)    

                print(f'calculate : {self.timestamp}')
                # for deg in self.positions_deg:
                #     print(deg)
                self.data.ctrl = self.positions_deg

                mj.mj_forward(self.model, self.data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

                force_fingers = np.array(self.sensordata) # 1x9
                force_fingers = force_fingers[0:3,0:3].flatten()


                ee_pos1 = np.subtract(goals[0], self.data.site("FFtip").xpos)
                ee_pos2 = np.subtract(goals[1], self.data.site("MFtip").xpos)
                ee_pos3 = np.subtract(goals[2], self.data.site("THtip").xpos)

                forces_list.append(force_fingers)
                ee_list.append(np.array([ee_pos1, ee_pos2, ee_pos3]).reshape(1,9)[0])
                
                #Step the simulation.
                mj.mj_step(self.model, self.data)
                viewer.sync()
                time = self.timestamp
                # else : 
                #     
        
            _, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
            forces_list = np.array(forces_list)
            ee_list = np.array(ee_list)
            
            # Saving the arrays to CSV files
            # np.savetxt('/home/hanlab/Desktop/work_dir/plot_result/force_list.csv', force_list, delimiter=',')
            np.savetxt('/home/hanlab/Desktop/work_dir/plot_result/forces_list.csv', forces_list, delimiter=',')
            np.savetxt('/home/hanlab/Desktop/work_dir/plot_result/ee_list.csv', ee_list, delimiter=',')
            
            # print(force_list)
            sim_time = range(time)
            lines = ax[0].plot(sim_time,forces_list, label='Force sensor')
            ax[0].set_title('Force sensor')
            ax[0].set_ylabel('Newtons')
            ax[0].set_xlabel('time')
            ax[0].legend(iter(lines), ('$F_x$', '$F_y$', '$F_z$'))

            print()
            lines = ax[1].plot(sim_time,ee_list, label='EE Error')
            if pd_flag:
                ax[1].set_title('Impedance controller')
            if pd_flag:
                ax[1].set_title('PD controller')
            ax[1].set_ylabel('Newtons')
            ax[1].set_xlabel('time')
            ax[1].legend(iter(lines), ('FFL10','FFL11','FFL12','MFL20','MFL21','MFL22','THL30','THL31','THL32'))
            plt.tight_layout()
            plt.show()

def read_write_py_node():
    rclpy.init(args=None)
    minimal_service = MinimalService()
    try :
        rclpy.spin(minimal_service)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        rclpy.shutdown()



def main():
    read_write_py_node()


if __name__ == '__main__':
    main()