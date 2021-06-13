import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import os


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def plot_env_step(img, step, name=None):
    filename = name + '.pdf'
    print(filename)
    with PdfPages(filename) as pdf:
        print(filename)
        plt.figure()
        plt.imshow(img)
        plt.title('Step %d' % step)
        pdf.savefig()
        plt.close()


def animate_env_obs(imgs, name):
    assert imgs is not None
    fps = 30
    n_seconds = 5
    fig = plt.figure(figsize=(8, 8))
    init_img = imgs[0]
    img = plt.imshow(init_img)

    def animate_func(i):
        img.set_array(imgs[i])
        return [img]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=n_seconds * fps,
        interval=1000 / fps,  # in ms
    )
    anim.save(name+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
