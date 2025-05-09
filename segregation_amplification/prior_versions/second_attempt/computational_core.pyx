import cython
import numpy as np
cimport numpy as np

DTYPEUINT8 = np.uint8
DTYPEBYTE = np.byte
DTYPEINT = np.int64
DTYPEINT64 = np.int64
DTYPELONG = np.long
DTYPEFLOAT32 = np.float32
DTYPEFLOAT64 = np.float64
ctypedef np.uint8_t DTYPEUINT8_t
ctypedef np.int_t DTYPEINT_t
ctypedef np.int64_t DTYPEINT64_t
ctypedef np.float32_t DTYPEFLOAT32_t
ctypedef np.float64_t DTYPEFLOAT64_t



@cython.cdivision(True)
@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(True)
cpdef float[::, ::1] spatial_externality_game(float[::, ::1] direct_benefit, float[::, ::1] social_benefit, float[::, ::1] reciprocal_response):
    cdef long n_rows = direct_benefit.shape[0]
    cdef long n_cols = direct_benefit.shape[1]
    cdef long r, c, rd, cd
    cdef float utility_from_masking, utility_from_not_masking

    cdef float[::, ::1] output_array = np.empty([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef float[::, ::1] masking_choice = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef float[::, ::1] average_reciprocal_response_from_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef float[::, ::1] average_reciprocal_response_from_not_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)


    for r in range(n_rows):
        for c in range(n_cols):

            output_array[r, c] = direct_benefit[r, c] + social_benefit[r, c] + reciprocal_response[r, c]

            for rd in range(-1, 1):
                for cd in range(-1, 1):
                    # Possible optimization: remove testing for own cell by smartly building tuples of which to test. Might also be done when considering non d8 neighbors.

                    if not (rd == 0 and cd == 0): # Make sure to not account for one's reciprocal response to themself.
                        average_reciprocal_response_from_masking[r, c] += reciprocal_response[r + rd, c + cd]
                        average_reciprocal_response_from_not_masking[r, c] += -1 * reciprocal_response[r + rd, c + cd]

            average_reciprocal_response_from_masking[r, c] = average_reciprocal_response_from_masking[r, c] / 8.
            utility_from_masking = direct_benefit[r, c] + social_benefit[r, c] * average_reciprocal_response_from_masking[r, c] # Note implicit masking_choice as * 1
            utility_from_not_masking = -1 * direct_benefit[r, c] + social_benefit[r, c] * average_reciprocal_response_from_not_masking[r, c]

            if utility_from_masking > utility_from_not_masking:
                masking_choice[r, c] = 1
            else:
                masking_choice[r, c] = -1

    return masking_choice

@cython.cdivision(True)
@cython.embedsignature(True)
@cython.boundscheck(True)
@cython.wraparound(True)
def spatial_segregation_game(long[::1] agent_ids,
                             long[::, ::1] agent_locations,
                             long[::, ::1] unoccupied_locations,
                             long[::1] agent_types,
                             long[::, ::1] agent_ids_map,
                             long[::, ::1] types_map,
                             float threshold,
                             str game_type,
                             int reporting_threshold=0):

    cdef long n_agents = len(agent_ids)
    cdef long n_rows = agent_ids_map.shape[0]
    cdef long n_cols = agent_ids_map.shape[1]
    cdef long n_cells = n_rows * n_cols
    cdef long iteration_counter, random_unoccupied_index
    cdef long r, c, rd, cd, neighborhood_r, neighborhood_c, new_r, new_c

    cdef float sum_mismatch, mean_mismatch, sum_match, mean_match, sum_metric, mean_metric
    cdef float utility_from_masking, utility_from_not_masking

    cdef long[::, ::1] occupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    cdef long[::, ::1] unoccupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    cdef float[::, ::1] mean_metric_map = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)

    cdef long n_unoccupied_cells = n_cells - n_agents

    cdef long n_changers = 0

    if game_type == 'move_if_not_enough_similar':
        for agent_id in range(n_agents):
            sum_metric = 0
            r = agent_locations[agent_id, 0]
            c = agent_locations[agent_id, 1]
            for rd in range(-1, 2):
                for cd in range(-1, 2):
                    neighborhood_r = r + rd
                    neighborhood_c = c + cd
                    if neighborhood_r >= n_rows:
                        neighborhood_r -= n_rows
                    if neighborhood_c >= n_cols:
                        neighborhood_c -= n_cols
                    if types_map[r, c] == types_map[neighborhood_r, neighborhood_c]:
                        if not (neighborhood_r == r and neighborhood_c == c): # Make sure to not account for one's reciprocal response to themself.
                            sum_metric += 1
            mean_metric = sum_metric / 8.0
            mean_metric_map[r, c] = mean_metric

            if mean_metric < threshold:
                n_changers += 1
                random_unoccupied_index = np.random.randint(0, n_unoccupied_cells)

                new_r = unoccupied_locations[random_unoccupied_index, 0]
                new_c = unoccupied_locations[random_unoccupied_index, 1]

                agent_locations[agent_id, 0] = new_r
                agent_locations[agent_id, 1] = new_c

                agent_ids_map[new_r, new_c] = agent_id
                agent_ids_map[r, c] = 0

                types_map[new_r, new_c] = types_map[r, c]
                types_map[r, c] = 0
                unoccupied_locations[random_unoccupied_index, 0] = r
                unoccupied_locations[random_unoccupied_index, 1] = c

    elif game_type == 'move_if_too_many_dissimilar':
        for agent_id in range(n_agents):
            sum_metric = 0
            r = agent_locations[agent_id, 0]
            c = agent_locations[agent_id, 1]
            for rd in range(-1, 2):
                for cd in range(-1, 2):
                    neighborhood_r = r + rd
                    neighborhood_c = c + cd
                    if neighborhood_r >= n_rows:
                        neighborhood_r -= n_rows
                    if neighborhood_c >= n_cols:
                        neighborhood_c -= n_cols
                    if types_map[r, c] != types_map[neighborhood_r, neighborhood_c] and types_map[neighborhood_r, neighborhood_c] != 0:
                        if not (neighborhood_r == r and neighborhood_c == c): # Make sure to not account for one's reciprocal response to themself.
                            sum_metric += 1

            mean_metric = sum_metric / 8.0
            mean_metric_map[r, c] = mean_metric

            if mean_metric > threshold:
                n_changers += 1
                random_unoccupied_index = np.random.randint(0, n_unoccupied_cells)

                new_r = unoccupied_locations[random_unoccupied_index, 0]
                new_c = unoccupied_locations[random_unoccupied_index, 1]

                agent_locations[agent_id, 0] = new_r
                agent_locations[agent_id, 1] = new_c

                agent_ids_map[new_r, new_c] = agent_id
                agent_ids_map[r, c] = 0

                types_map[new_r, new_c] = types_map[r, c]
                types_map[r, c] = 0
                unoccupied_locations[random_unoccupied_index, 0] = r
                unoccupied_locations[random_unoccupied_index, 1] = c

    if reporting_threshold > 0:
        print('number of agents who moved: ' + str(n_changers))

    return n_changers




@cython.cdivision(True)
@cython.embedsignature(True)
@cython.boundscheck(True)
@cython.wraparound(True)
def spatial_segregation_externality_game(long[::1] agent_ids,
                                         long[::, ::1] agent_locations,
                                         long[::, ::1] unoccupied_locations,
                                         long[::1] agent_types,
                                         long[::, ::1] agent_ids_map,
                                         long[::, ::1] types_map,
                                         float threshold,
                                         long neighborhood_radius,
                                         str game_type,
                                         float[::, ::1] direct_benefit,
                                         float[::, ::1] social_benefit,
                                         float[::, ::1] reciprocal_response,
                                         float[::, ::1] masking_choice,
                                         float[::, ::1] average_reciprocal_response_from_masking,
                                         float[::, ::1] mean_metric_map,
                                         long reporting_threshold=0):

    cdef long n_agents = len(agent_ids)
    cdef long n_rows = agent_ids_map.shape[0]
    cdef long n_cols = agent_ids_map.shape[1]
    cdef long n_cells = n_rows * n_cols
    cdef long iteration_counter, random_unoccupied_index
    cdef long r, c, rd, cd, neighborhood_r, neighborhood_c, new_r, new_c

    cdef float sum_mismatch, mean_mismatch, sum_match, mean_match, sum_metric, mean_metric, n_in_neighborhood
    cdef float utility_from_masking, utility_from_not_masking

    cdef long[::, ::1] occupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    cdef long[::, ::1] unoccupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    # cdef float[::, ::1] mean_metric_map = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)

    cdef long n_unoccupied_cells = n_cells - n_agents

    cdef long n_changers = 0

    if game_type == 'move_if_not_enough_similar':
        for agent_id in range(n_agents):

            sum_metric = 0
            r = agent_locations[agent_id, 0]
            c = agent_locations[agent_id, 1]
            for rd in range(-neighborhood_radius, neighborhood_radius + 1):
                for cd in range(-neighborhood_radius, neighborhood_radius + 1):
                    neighborhood_r = r + rd
                    neighborhood_c = c + cd
                    if neighborhood_r >= n_rows:
                        neighborhood_r -= n_rows
                    if neighborhood_c >= n_cols:
                        neighborhood_c -= n_cols

                    average_reciprocal_response_from_masking[r, c] += reciprocal_response[neighborhood_r, neighborhood_c]

                    if types_map[r, c] == types_map[neighborhood_r, neighborhood_c]:
                        if not (neighborhood_r == r and neighborhood_c == c): # Make sure to not account for one's reciprocal response to themself.
                            sum_metric += 1

            n_in_neighborhood = <float>(neighborhood_radius * 2 + 1) ** 2 - 1
            mean_metric = sum_metric / n_in_neighborhood
            mean_metric_map[r, c] = mean_metric

            average_reciprocal_response_from_masking[r, c] = average_reciprocal_response_from_masking[r, c] / n_in_neighborhood
            utility_from_masking = direct_benefit[r, c] + social_benefit[r, c] * average_reciprocal_response_from_masking[r, c] # Note implicit masking_choice as * 1
            utility_from_not_masking = -1 * direct_benefit[r, c] + -1 * social_benefit[r, c] * average_reciprocal_response_from_masking[r, c]

            if utility_from_masking > utility_from_not_masking:
                masking_choice[r, c] = 1.
            else:
                masking_choice[r, c] = -1.


            if mean_metric < threshold:
                n_changers += 1
                random_unoccupied_index = np.random.randint(0, n_unoccupied_cells)

                new_r = unoccupied_locations[random_unoccupied_index, 0]
                new_c = unoccupied_locations[random_unoccupied_index, 1]

                # if np.random.rand() < .001:
                #     print('found agent ' + str(agent_ids_map[r, c]) + ' with params ' + str(types_map[r, c]) + ' ' +
                #         str(direct_benefit[r, c]) + ' ' +
                #         str(social_benefit[r, c]) + ' ' +
                #         str(masking_choice[r, c]) + ' ')

                agent_locations[agent_id, 0] = new_r
                agent_locations[agent_id, 1] = new_c

                agent_ids_map[new_r, new_c] = agent_id
                agent_ids_map[r, c] = 0

                types_map[new_r, new_c] = types_map[r, c]
                types_map[r, c] = 0

                direct_benefit[new_r, new_c] = direct_benefit[r, c]
                direct_benefit[r, c] = 0

                social_benefit[new_r, new_c] = social_benefit[r, c]
                social_benefit[r, c] = 0
                reciprocal_response[new_r, new_c] = reciprocal_response[r, c]
                reciprocal_response[r, c] = 0

                masking_choice[new_r, new_c] = masking_choice[r, c]
                masking_choice[r, c] = 0

                average_reciprocal_response_from_masking[new_r, new_c] = average_reciprocal_response_from_masking[r, c]
                average_reciprocal_response_from_masking[r, c] = 0

                unoccupied_locations[random_unoccupied_index, 0] = r
                unoccupied_locations[random_unoccupied_index, 1] = c

    elif game_type == 'move_if_too_many_dissimilar':
        for agent_id in range(n_agents):

            sum_metric = 0
            r = agent_locations[agent_id, 0]
            c = agent_locations[agent_id, 1]
            for rd in range(-neighborhood_radius, neighborhood_radius + 1):
                for cd in range(-neighborhood_radius, neighborhood_radius + 1):
                    neighborhood_r = r + rd
                    neighborhood_c = c + cd
                    if neighborhood_r >= n_rows:
                        neighborhood_r -= n_rows
                    if neighborhood_c >= n_cols:
                        neighborhood_c -= n_cols

                    average_reciprocal_response_from_masking[r, c] += reciprocal_response[neighborhood_r, neighborhood_c]

                    if types_map[r, c] != types_map[neighborhood_r, neighborhood_c] and types_map[neighborhood_r, neighborhood_c] != 0:
                        if not (neighborhood_r == r and neighborhood_c == c):  # Make sure to not account for one's reciprocal response to themself.
                            sum_metric += 1

            n_in_neighborhood = <float> (neighborhood_radius * 2 + 1) ** 2 - 1
            mean_metric = sum_metric / n_in_neighborhood
            mean_metric_map[r, c] = mean_metric

            average_reciprocal_response_from_masking[r, c] = average_reciprocal_response_from_masking[r, c] / n_in_neighborhood
            utility_from_masking = direct_benefit[r, c] + social_benefit[r, c] * average_reciprocal_response_from_masking[r, c]  # Note implicit masking_choice as * 1
            utility_from_not_masking = -1 * direct_benefit[r, c] + -1 * social_benefit[r, c] * average_reciprocal_response_from_masking[r, c]

            if utility_from_masking > utility_from_not_masking:
                masking_choice[r, c] = 1.
            else:
                masking_choice[r, c] = -1.

            if mean_metric > threshold:
                n_changers += 1
                random_unoccupied_index = np.random.randint(0, n_unoccupied_cells)

                new_r = unoccupied_locations[random_unoccupied_index, 0]
                new_c = unoccupied_locations[random_unoccupied_index, 1]

                # if np.random.rand() < .001:
                #     print('found agent ' + str(agent_ids_map[r, c]) + ' with params ' + str(types_map[r, c]) + ' ' +
                #         str(direct_benefit[r, c]) + ' ' +
                #         str(social_benefit[r, c]) + ' ' +
                #         str(masking_choice[r, c]) + ' ')

                agent_locations[agent_id, 0] = new_r
                agent_locations[agent_id, 1] = new_c

                agent_ids_map[new_r, new_c] = agent_id
                agent_ids_map[r, c] = 0

                types_map[new_r, new_c] = types_map[r, c]
                types_map[r, c] = 0

                direct_benefit[new_r, new_c] = direct_benefit[r, c]
                direct_benefit[r, c] = 0

                social_benefit[new_r, new_c] = social_benefit[r, c]
                social_benefit[r, c] = 0
                reciprocal_response[new_r, new_c] = reciprocal_response[r, c]
                reciprocal_response[r, c] = 0

                masking_choice[new_r, new_c] = masking_choice[r, c]
                masking_choice[r, c] = 0

                average_reciprocal_response_from_masking[new_r, new_c] = average_reciprocal_response_from_masking[r, c]
                average_reciprocal_response_from_masking[r, c] = 0

                unoccupied_locations[random_unoccupied_index, 0] = r
                unoccupied_locations[random_unoccupied_index, 1] = c




    if reporting_threshold > 0:
        print('number of agents who moved: ' + str(n_changers) + ' with average masking of ' + str(np.mean(masking_choice)))

    return n_changers



@cython.cdivision(True)
@cython.embedsignature(True)
@cython.boundscheck(True)
@cython.wraparound(True)
def spatial_segregation_externality_game_update_masking_params(long[::1] agent_ids,
                                         long[::, ::1] agent_locations,
                                         long[::, ::1] unoccupied_locations,
                                         long[::1] agent_types,
                                         long[::, ::1] agent_ids_map,
                                         long[::, ::1] types_map,
                                         float threshold,
                                         long neighborhood_radius,
                                         str game_type,
                                         float[::, ::1] direct_benefit,
                                         float[::, ::1] social_benefit,
                                         float[::, ::1] reciprocal_response,
                                         float[::, ::1] masking_choice,
                                         float[::, ::1] average_reciprocal_response_from_masking,
                                         float[::, ::1] mean_metric_map,
                                         long reporting_threshold=0):

    cdef long n_agents = len(agent_ids)
    cdef long n_rows = agent_ids_map.shape[0]
    cdef long n_cols = agent_ids_map.shape[1]
    cdef long n_cells = n_rows * n_cols
    cdef long iteration_counter, random_unoccupied_index
    cdef long r, c, rd, cd, neighborhood_r, neighborhood_c, new_r, new_c

    cdef float sum_mismatch, mean_mismatch, sum_match, mean_match, sum_metric, mean_metric, n_in_neighborhood
    cdef float utility_from_masking, utility_from_not_masking

    cdef long[::, ::1] occupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    cdef long[::, ::1] unoccupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    # cdef float[::, ::1] mean_metric_map = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)

    cdef long n_unoccupied_cells = n_cells - n_agents

    cdef long n_changers = 0

    for agent_id in range(n_agents):

        r = agent_locations[agent_id, 0]
        c = agent_locations[agent_id, 1]
        for rd in range(-neighborhood_radius, neighborhood_radius + 1):
            for cd in range(-neighborhood_radius, neighborhood_radius + 1):
                neighborhood_r = r + rd
                neighborhood_c = c + cd
                if neighborhood_r >= n_rows:
                    neighborhood_r -= n_rows
                if neighborhood_c >= n_cols:
                    neighborhood_c -= n_cols

                average_reciprocal_response_from_masking[r, c] += reciprocal_response[neighborhood_r, neighborhood_c]

                if types_map[r, c] == types_map[neighborhood_r, neighborhood_c]:
                    if not (neighborhood_r == r and neighborhood_c == c): # Make sure to not account for one's reciprocal response to themself.
                        sum_metric += 1

        n_in_neighborhood = <float>(neighborhood_radius * 2 + 1) ** 2 - 1
        mean_metric = sum_metric / n_in_neighborhood
        mean_metric_map[r, c] = mean_metric

        average_reciprocal_response_from_masking[r, c] = average_reciprocal_response_from_masking[r, c] / n_in_neighborhood
        utility_from_masking = direct_benefit[r, c] + social_benefit[r, c] * average_reciprocal_response_from_masking[r, c] # Note implicit masking_choice as * 1
        utility_from_not_masking = -1 * direct_benefit[r, c] + -1 * social_benefit[r, c] * average_reciprocal_response_from_masking[r, c]

        if utility_from_masking > utility_from_not_masking:
            masking_choice[r, c] = 1.
        else:
            masking_choice[r, c] = -1.

    if reporting_threshold > 0:
        print('number of agents who moved: ' + str(n_changers) + ' with average masking of ' + str(np.mean(masking_choice)))

    return n_changers