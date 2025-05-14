import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment(seed=0):

    np.random.seed(seed)

    # # example --> you can add as many datasets as you want in the lists!
    env = ImitationFactory.make("UnitreeH1",
                                # if SMPL and AMASS are installed, you can use the following:
                                amass_dataset_conf=AMASSDatasetConf(["ACCAD/ACCAD/Female1General_c3d/A12 - crawl backwards_poses.npz"]),
                                n_substeps=20)

    env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


if __name__ == '__main__':
    experiment()
