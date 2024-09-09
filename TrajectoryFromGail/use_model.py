import pickle

import gym
import numpy as np
import tensorflow as tf
import argparse
from network_models.policy_net import Policy_net
from infrastructure.environment_wrapper import EnvironmentWrapper
from infrastructure.utils import open_file_and_save
env_wrapper = EnvironmentWrapper()
def get_trajectory(image, point1, point2, display):
    tf.compat.v1.disable_eager_execution()
    from realpic_env import GridWorldEnv
    # PS:输入图像和两个端点坐标
    env = GridWorldEnv(img_path=image, up=point1, down=point2)
    env.seed(0)
    np.random.seed(2)
    Policy = Policy_net('policy', env)
    saver = tf.compat.v1.train.Saver()
    render = False
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, f"trained_model/final_multi/model_CartPole-v0.ckpt")
        obs = env.reset()
        reward = 0
        success_num = 0
        frames = []
        for iteration in range(1000):
            rewards = []
            data = []
            trajactory = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, _ = Policy.act(obs=obs, stochastic=True)
                act = np.ndarray.item(act)
                rewards.append(reward)
                next_obs, reward, done, info = env.step(act)
                if render:
                    if display:
                        frames.append(env.render(mode='rgb_array'))
                    data.append((obs, act, reward))
                    trajactory.append((int(env.n_width-env.position[0]-1)*3+3//2, int(env.position[1])*3+3//2))
                if done:
                    obs = env.reset()
                    break
                else:
                    obs = next_obs
            if render:
                break
            if sum(rewards) >= 40:
                success_num += 1
                render = True
            else:
                success_num = 0
        if iteration == 999:
            print("Failed to find the trajectory!")
        if len(frames) > 1 and display:
            display_frames_as_gif(frames)
        return(trajactory)

# 用于保存最终运行结果GIF的方法
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def display_frames_as_gif(frames):
    # 创建一个新的figure对象，并设置其大小
    fig, ax = plt.subplots(figsize=(2, 6))  # 这里的figsize参数设置了画布的大小
    # 清除坐标轴
    ax.axis('off')
    # 初始化图像显示
    patch = ax.imshow(frames[0], animated=True)  # 添加animated=True以确保动画有效

    def animate(i):
        patch.set_data(frames[i])
        # 使用fig（而不是plt.gcf()）作为动画的figure
    anim = FuncAnimation(fig, animate, frames=range(len(frames)),
                         interval=50)  # 注意这里frames参数通常是一个可迭代对象，这里使用range(len(frames))
    # 保存GIF动画
    anim.save('./result.gif', writer='pillow', fps=10)


if __name__ == '__main__':
    # Added for compatibility with Tensorflow 2.x
    image_xi = "xi.jpg"
    up = (758, 320)   #758,320, 912-758, 352-320
    down = (912, 352)
    display = False
    get_trajectory(image_xi, up, down, display=display)
