import os
import sys
import random
import numpy as np
from numpy.core import multiarray
from numpy.core.defchararray import upper
import scipy as sp
from scipy.stats import truncnorm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import colors
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer or Qt5Agg

sys.path.append('C:/Files/Research/hazelbean/hazelbean_dev/')
import hazelbean as hb

p = hb.ProjectFlow(r"C:\Files\Research\abm\tribal_masking\projects\first_attempt")

L = hb.get_logger('tribal_masking')


import tribal_masking_computational_core

def initialize_agents(spatial_shape, proportion_filled, type_bias, params):
    ### INITIALIZATION
    # Save row and col ids as two new parameters so that we can quickly look up paraeters my spatial location OR agent id.
    row_ids = np.arange(spatial_shape[0])
    col_ids = np.arange(spatial_shape[1])

    # Number of agents is a function then of the size of the space and the proportion filled, rounded down and inted.
    n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))
    n_cells = spatial_shape[0] * spatial_shape[1]

    # Keep track of all the agents via IDs
    agent_ids = np.arange(0, n_agents).astype(np.int64)

    # positional_indices records the 2-length r, c of each agent grid-cell based on their r, c.
    positional_indices = np.empty((spatial_shape[0], spatial_shape[1], 2), dtype=int)
    positional_indices[:, :, 0] = row_ids[:, None]
    positional_indices[:, :, 1] = col_ids

    # Somewhat convoluted way of randomizing the initial postitions.
    shuffled_positional_indices = np.copy(positional_indices)  # Operate on a copy to keep original

    # Shuffle requires a flattened array, so reshape it from 3d to 2d, keeping the 2d r-c indices.
    new_shape = (shuffled_positional_indices.shape[0] * shuffled_positional_indices.shape[1], shuffled_positional_indices.shape[2])
    shuffled_positional_indices = shuffled_positional_indices.reshape(new_shape)

    # Randomize the order by creating a randomized list of r-c indices for future fast iterating.
    # LEARNING POINT, this was the most efficient approach I could think of for shuffling in the first 2 of three directions while keeping the third intact.
    rng = np.random.default_rng()
    rng.shuffle(shuffled_positional_indices, axis=0)

    # First initialize all agents_list on a random location according to the initial shuffled queue.
    agent_locations = shuffled_positional_indices[:n_agents]

    # Also initialize the unoccupied locations. When agents move, the grab a new spot from this list and move their previous slot into this array.
    unoccupied_locations = shuffled_positional_indices[n_agents:]

    # Agent types are 1 = liberal, 2 = conservative. 0 is left blank to reflect absence of agents.
    agent_types = np.random.randint(1, 3, size=n_agents).astype(np.int64)

    # agent ids and types are initialized as maps.
    agent_ids_map = np.zeros((spatial_shape[0], spatial_shape[1]), dtype=np.int64)
    types_map = np.zeros((spatial_shape[0], spatial_shape[1]), dtype=np.int64)
    for i in range(n_agents):
        agent_ids_map[shuffled_positional_indices[i, 0], shuffled_positional_indices[i, 1]] = i
        types_map[shuffled_positional_indices[i, 0], shuffled_positional_indices[i, 1]] = agent_types[i]

    # hb.show(types_map)
    a = sp.stats.truncnorm.rvs((params['d'][0] - params['d'][2]) / params['d'][3], (params['d'][1] - params['d'][2]) / params['d'][3], loc=params['d'][2], scale=params['d'][3], size=n_cells)
    a = np.where(agent_ids_map.flatten() > 0, a, 0.)
    d_params = a.reshape(spatial_shape).astype(np.float32)
    d = np.copy(d_params)

    a = sp.stats.truncnorm.rvs((params['s'][0] - params['s'][2]) / params['s'][3], (params['s'][1] - params['s'][2]) / params['s'][3], loc=params['s'][2], scale=params['s'][3], size=n_cells)
    a = np.where(agent_ids_map.flatten() > 0, a, 0.)
    s_params = a.reshape(spatial_shape).astype(np.float32)
    s = np.copy(s_params)

    a = sp.stats.truncnorm.rvs((params['r'][0] - params['r'][2]) / params['r'][3], (params['r'][1] - params['r'][2]) / params['r'][3], loc=params['r'][2], scale=params['r'][3], size=n_cells)
    a = np.where(agent_ids_map.flatten() > 0, a, 0.)
    add_type_corellation = 1
    if add_type_corellation:
        a = np.where(types_map.flatten() == 2, a * type_bias, a)
    r_initial = a.reshape(spatial_shape).astype(np.float32)
    r = np.copy(r_initial)

    return agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r


def get_masking_choice_full_resolve(spatial_shape, proportion_filled, game_type, threshold, neighborhood_radius, type_bias, d_param, dd_param, s_param, sd_param, r_param, rd_param):
    # For convenience, we define input parameters here via a dictionary.
    params = {'d': [-10, 10, d_param, dd_param], 's': [0, 1, s_param, sd_param], 'r': [-10, 10, r_param, rd_param]}
    # params = {'d': [-10, 10, -.25, .05], 's': [0, 1, 0.9, .1], 'r': [-10, 10, 1.0, 1.95]}
    masking_choice = np.zeros(spatial_shape).astype(np.float32)
    mean_metric_map = np.zeros(spatial_shape).astype(np.float32)
    average_reciprocal_response_from_masking = np.zeros(spatial_shape).astype(np.float32)
    agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r = \
        initialize_agents(spatial_shape, proportion_filled, type_bias, params)
    neighborhood_radius = np.int64(neighborhood_radius)
    n_iterations = 150
    for i in range(n_iterations):

        n_changed = tribal_masking_computational_core.spatial_segregation_externality_game(
            agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
            types_map, threshold, neighborhood_radius, game_type,
            d, s, r, masking_choice, average_reciprocal_response_from_masking, mean_metric_map,
            reporting_threshold=0)
        # End early if few agents move locations.
        min_changers_to_end = 1
        if n_changed <= min_changers_to_end:
            break

    return np.copy(np.asarray(masking_choice)), np.copy(np.asarray(types_map)), np.copy(np.asarray(mean_metric_map)), np.copy(np.asarray(average_reciprocal_response_from_masking))


def plot_combined_game_interactive_full_resolve(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init):


    axis_color = 'lightgoldenrodyellow'

    fig = plt.figure()
    ax_ul = fig.add_subplot(221)
    ax_ur = fig.add_subplot(222)
    ax_ll = fig.add_subplot(223)
    ax_lr = fig.add_subplot(224)

    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(bottom=0.4)
    col_dict = {-1: "purple",
                0: "white",
                1: "green", }
    labels = np.array(["No mask", "Vacant", "Mask"])

    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ul = ax_ul.imshow(a1, cmap=cm, norm=norm, interpolation='nearest')
    cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)

    col_dict = {
        0: "white",
        1: "blue",
        2: "red"
    }
    labels = np.array(["Vacant", "Tribe 1", "Tribe 2"])
    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ur = ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
    cb_ur = fig.colorbar(im_ur, ax=ax_ur, format=None, ticks=tickz)

    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)

    # im_ul = ax_ul.imshow(a1, cmap='PRGn', interpolation='nearest', vmin=-1.5, vmax=1.5)
    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    # im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    # im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)
    #
    percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
    percent_masked_annotation = ax_ul.annotate('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)), xy=(0.01, 0.95), xycoords='figure fraction')

    # diff = norm_bins[1:] - norm_bins[:-1]
    # tickz = norm_bins[:-1] + diff / 2
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, ticks=[-1, 1], shrink=.8 )
    # cb_ul.ax.set_yticklabels(['no mask', 'mask'])

    # fig.colorbar(im_ur, ax=ax_ur, shrink=.8)

    fig.colorbar(im_ll, ax=ax_ll, shrink=.8)
    fig.colorbar(im_lr, ax=ax_lr, shrink=.8, extend='both')

    ax_ul.set_title('Masking choice')
    ax_ur.set_title('Political type')
    ax_ll.set_title('Neighborhood similarity')
    ax_lr.set_title('Neighborhood r')

    ax_ul.axis('off')
    ax_ur.axis('off')
    ax_ll.axis('off')
    ax_lr.axis('off')

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        a1, a2, a3, a4 = get_masking_choice_full_resolve(spatial_shape, proportion_filled, game_type, threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val)
        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a1)
        im_ur.set_data(a2)
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()

    # Turn off hover text (was breaking the window size)
    ax_ul.format_coord = lambda x, y: ""
    ax_ur.format_coord = lambda x, y: ""
    ax_ll.format_coord = lambda x, y: ""
    ax_lr.format_coord = lambda x, y: ""

    # Add Sliders
    # Define an axes area and draw a1 slider in it
    threshold_slider_ax = fig.add_axes([.3, 0.27, 0.4, 0.03], facecolor=axis_color)
    threshold_slider = Slider(threshold_slider_ax, 'threshold', 0.0, 1.0, valinit=threshold_init)
    threshold_slider.on_changed(sliders_on_changed)

    neighborhood_radius_slider_ax = fig.add_axes([.3, 0.24, 0.4, 0.03], facecolor=axis_color)
    neighborhood_radius_slider = Slider(neighborhood_radius_slider_ax, 'neighborhood_radius', 1., 6., valinit=neighborhood_radius_init)
    neighborhood_radius_slider.on_changed(sliders_on_changed)

    type_bias_slider_ax = fig.add_axes([.3, 0.21, 0.4, 0.03], facecolor=axis_color)
    type_bias_slider = Slider(type_bias_slider_ax, 'type_bias', -2., 2., valinit=type_bias_init)
    type_bias_slider.on_changed(sliders_on_changed)

    d_slider_ax = fig.add_axes([0.3, 0.18, 0.4, 0.03], facecolor=axis_color)
    d_slider = Slider(d_slider_ax, 'd', -5., 5., valinit=d_init)
    d_slider.on_changed(sliders_on_changed)

    dd_slider_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03], facecolor=axis_color)
    dd_slider = Slider(dd_slider_ax, 'dd', 0.00001, 5., valinit=dd_init)
    dd_slider.on_changed(sliders_on_changed)

    s_slider_ax = fig.add_axes([0.3, 0.12, 0.4, 0.03], facecolor=axis_color)
    s_slider = Slider(s_slider_ax, 's', .0000001, 4., valinit=s_init)
    s_slider.on_changed(sliders_on_changed)

    sd_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.03], facecolor=axis_color)
    sd_slider = Slider(sd_slider_ax, 'sd', 0.0000001, 1., valinit=sd_init)
    sd_slider.on_changed(sliders_on_changed)

    r_slider_ax = fig.add_axes([0.3, 0.06, 0.4, 0.03], facecolor=axis_color)
    r_slider = Slider(r_slider_ax, 'r', -5, 5, valinit=r_init)
    r_slider.on_changed(sliders_on_changed)

    rd_slider_ax = fig.add_axes([0.3, .03, 0.4, 0.03], facecolor=axis_color)
    rd_slider = Slider(rd_slider_ax, 'rd', 0.0000001, 1., valinit=rd_init)
    rd_slider.on_changed(sliders_on_changed)

    plt.show()

def plot_combined_game_interactive_time_variant(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init):
    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed_just_masking_updates(val):
        params = {'d': [-10, 10, d_slider.val, dd_slider.val], 's': [0, 1, s_slider.val, sd_slider.val], 'r': [-10, 10, r_slider.val, rd_slider.val]}
        # params_init = {'d': [-10, 10, -.25, .0000001], 's': [0, 10, 1.0, .0000001], 'r': [-10, 10, 3.0, .0000001]}
        # d_init = params_init['d'][2]
        # dd_init = params_init['d'][3]
        # s_init = params_init['s'][2]
        # sd_init = params_init['s'][3]
        # r_init = params_init['r'][2]
        # rd_init = params_init['r'][3]

        row_ids = np.arange(spatial_shape[0])
        col_ids = np.arange(spatial_shape[1])

        # Number of agents is a function then of the size of the space and the proportion filled, rounded down and inted.
        n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))
        n_cells = spatial_shape[0] * spatial_shape[1]

        a = sp.stats.truncnorm.rvs((params['d'][0] - params['d'][2]) / params['d'][3], (params['d'][1] - params['d'][2]) / params['d'][3], loc=params['d'][2], scale=params['d'][3], size=n_cells)
        a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        d_paramsial = a.reshape(spatial_shape).astype(np.float32)
        d = np.copy(d_paramsial)

        a = sp.stats.truncnorm.rvs((params['s'][0] - params['s'][2]) / params['s'][3], (params['s'][1] - params['s'][2]) / params['s'][3], loc=params['s'][2], scale=params['s'][3], size=n_cells)
        a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        s_paramsial = a.reshape(spatial_shape).astype(np.float32)
        s = np.copy(s_paramsial)

        a = sp.stats.truncnorm.rvs((params['r'][0] - params['r'][2]) / params['r'][3], (params['r'][1] - params['r'][2]) / params['r'][3], loc=params['r'][2], scale=params['r'][3], size=n_cells)
        a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        a = np.where(types_map.flatten() == 2, a * type_bias_slider.val, a)
        r_initial = a.reshape(spatial_shape).astype(np.float32)
        r = np.copy(r_initial)

        a, a2, a3, a4 = update_masking_choice_without_reposition(threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val,
                                                                 agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r)
        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a)
        im_ur.set_data(a2)
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a == 1, 1, 0)) / n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()

    axis_color = 'lightgoldenrodyellow'

    fig = plt.figure()
    ax_ul = fig.add_subplot(221)
    ax_ur = fig.add_subplot(222)
    ax_ll = fig.add_subplot(223)
    ax_lr = fig.add_subplot(224)

    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(bottom=0.4)
    col_dict = {-1: "purple",
                0: "white",
                1: "green", }
    labels = np.array(["No mask", "Vacant", "Mask"])

    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ul = ax_ul.imshow(a1, cmap=cm, norm=norm, interpolation='nearest')
    cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)

    col_dict = {
        0: "white",
        1: "blue",
        2: "red"
    }
    labels = np.array(["Vacant", "Tribe 1", "Tribe 2"])
    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ur = ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
    cb_ur = fig.colorbar(im_ur, ax=ax_ur, format=None, ticks=tickz)

    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)

    # im_ul = ax_ul.imshow(a1, cmap='PRGn', interpolation='nearest', vmin=-1.5, vmax=1.5)
    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    # im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    # im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)
    #
    percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
    percent_masked_annotation = ax_ul.annotate('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)), xy=(0.01, 0.95), xycoords='figure fraction')

    # diff = norm_bins[1:] - norm_bins[:-1]
    # tickz = norm_bins[:-1] + diff / 2
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, ticks=[-1, 1], shrink=.8 )
    # cb_ul.ax.set_yticklabels(['no mask', 'mask'])

    # fig.colorbar(im_ur, ax=ax_ur, shrink=.8)

    fig.colorbar(im_ll, ax=ax_ll, shrink=.8)
    fig.colorbar(im_lr, ax=ax_lr, shrink=.8, extend='both')

    ax_ul.set_title('Masking choice')
    ax_ur.set_title('Political type')
    ax_ll.set_title('Neighborhood similarity')
    ax_lr.set_title('Neighborhood r')

    ax_ul.axis('off')
    ax_ur.axis('off')
    ax_ll.axis('off')
    ax_lr.axis('off')

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        a1, a2, a3, a4 = get_masking_choice_full_resolve(spatial_shape, proportion_filled, game_type, threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val)
        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a1)
        im_ur.set_data(a2)
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()

    # Turn off hover text (was breaking the window size)
    ax_ul.format_coord = lambda x, y: ""
    ax_ur.format_coord = lambda x, y: ""
    ax_ll.format_coord = lambda x, y: ""
    ax_lr.format_coord = lambda x, y: ""

    # Add Sliders
    # Define an axes area and draw a1 slider in it
    threshold_slider_ax = fig.add_axes([.3, 0.27, 0.4, 0.03], facecolor=axis_color)
    threshold_slider = Slider(threshold_slider_ax, 'threshold', 0.0, 1.0, valinit=threshold_init)
    threshold_slider.on_changed(sliders_on_changed)

    neighborhood_radius_slider_ax = fig.add_axes([.3, 0.24, 0.4, 0.03], facecolor=axis_color)
    neighborhood_radius_slider = Slider(neighborhood_radius_slider_ax, 'neighborhood_radius', 1., 6., valinit=neighborhood_radius_init)
    neighborhood_radius_slider.on_changed(sliders_on_changed)

    type_bias_slider_ax = fig.add_axes([.3, 0.21, 0.4, 0.03], facecolor=axis_color)
    type_bias_slider = Slider(type_bias_slider_ax, 'type_bias', -2., 2., valinit=type_bias_init)
    type_bias_slider.on_changed(sliders_on_changed)

    d_slider_ax = fig.add_axes([0.3, 0.18, 0.4, 0.03], facecolor=axis_color)
    d_slider = Slider(d_slider_ax, 'd', -5., 5., valinit=d_init)
    d_slider.on_changed(sliders_on_changed)

    dd_slider_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03], facecolor=axis_color)
    dd_slider = Slider(dd_slider_ax, 'dd', 0.00001, 5., valinit=dd_init)
    dd_slider.on_changed(sliders_on_changed)

    s_slider_ax = fig.add_axes([0.3, 0.12, 0.4, 0.03], facecolor=axis_color)
    s_slider = Slider(s_slider_ax, 's', .0000001, 4., valinit=s_init)
    s_slider.on_changed(sliders_on_changed)

    sd_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.03], facecolor=axis_color)
    sd_slider = Slider(sd_slider_ax, 'sd', 0.0000001, 1., valinit=sd_init)
    sd_slider.on_changed(sliders_on_changed)

    r_slider_ax = fig.add_axes([0.3, 0.06, 0.4, 0.03], facecolor=axis_color)
    r_slider = Slider(r_slider_ax, 'r', -5, 5, valinit=r_init)
    r_slider.on_changed(sliders_on_changed)

    rd_slider_ax = fig.add_axes([0.3, .03, 0.4, 0.03], facecolor=axis_color)
    rd_slider = Slider(rd_slider_ax, 'rd', 0.0000001, 1., valinit=rd_init)
    rd_slider.on_changed(sliders_on_changed)

    plt.show()


if __name__=='__main__':
    print ('You probably meant to run tribal_masking_main.py')