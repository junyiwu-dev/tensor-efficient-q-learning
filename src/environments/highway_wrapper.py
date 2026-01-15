import gymnasium as gym
import highway_env  # 确保导入以注册环境
import numpy as np

class HighwayEnvWrapper(gym.Wrapper):
    """包装器类，用于处理状态扁平化和API兼容性"""
    
    def __init__(self, env):
        super().__init__(env)
        # 更新观测空间为扁平化的9维向量
        self.observation_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(9,), 
            dtype=np.float32
        )
        
    def reset(self, **kwargs):
        """重置环境并扁平化观测"""
        result = self.env.reset(**kwargs)
        
        # 处理新版Gymnasium API返回值
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
            
        # 扁平化观测从(3,3)到(9,)
        obs_flat = obs.flatten() if isinstance(obs, np.ndarray) else np.array(obs).flatten()
        
        # 归一化到[-1, 1]范围（如果需要）
        # 基于配置中的features_range进行归一化
        obs_norm = np.clip(obs_flat / 100.0, -1.0, 1.0)  # 简单归一化
        
        return obs_norm  # 只返回观测，不返回info
    
    def step(self, action):
        """执行动作并扁平化观测"""
        result = self.env.step(action)
        
        # 处理新版Gymnasium API（5个返回值）
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated  # 合并为单个done标志
        elif len(result) == 4:
            obs, reward, done, info = result
        else:
            raise ValueError(f"Unexpected number of return values: {len(result)}")
        
        # 扁平化观测从(3,3)到(9,)
        obs_flat = obs.flatten() if isinstance(obs, np.ndarray) else np.array(obs).flatten()
        
        # 归一化到[-1, 1]范围
        obs_norm = np.clip(obs_flat / 100.0, -1.0, 1.0)
        
        # 返回4个值以保持与旧代码的兼容性
        return obs_norm, reward, done, info


class CustomHighwayEnv:
    """
    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    """
    def __new__(cls):
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 3,
                "features": ["x", "y", "vx"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "normalize": False,
                "order": "sorted"
            },
            "lanes_count": 3,
            "vehicles_count": 10,
            "simulation_frequency": 5,
            "duration": 50,
            "collision_reward": -1.0,
            "right_lane_reward": 0,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0,
            "reward_speed_range": [10, 30],
            "normalize_reward": False,
            "disable_collision_checks": False,  # 改为False以检测碰撞
            "initial_lane_id": 0,
            "vehicle_density": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "ego_spacing": 2
        }

        # 创建环境并传递配置
        env = gym.make("highway-v0", config=config)
        
        # 使用包装器处理状态扁平化和API兼容性
        wrapped_env = HighwayEnvWrapper(env)
        
        return wrapped_env
