

def animation_STC(time_series,name='generative'):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import HTML

    T = time_series.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.zeros((5,5)), aspect='auto',animated=True)
    im.set_clim(vmin=0, vmax=2)
    # plt.colorbar(im)

    def init():
        im.set_array(np.ma.array(np.ones((5,5)), mask=True))
        return im,

    def update(i):
        im.set_array(time_series[i,:,:])
        ax.set_title('t = ' + str(i))
        return im,

    ani = animation.FuncAnimation(fig, update, frames=range(T), init_func=init, interval=1000, blit=True)
    HTML(ani.to_jshtml())

    filename = name + '.gif'
    ani.save(filename, writer='imagemagick', fps=1)


# TODO: Code to draw plots given a graph class, do it in jupyter notebook

# TODO: plot marginal probabilities
# TODO: plot transitional probabilities

def graph_comparison(generative_graph, learned_graph):
    return

def KL_divergence(generative_model, learned_model):
    """plot KL divergence"""
    return
