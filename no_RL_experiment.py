import numpy as np

def penalty_deon(num_on_track, action, boosted = False):
    if action == 'switch':
        if boosted:
            return - 10
        else:
            return -1
    else:
        return 0

def penalty_util(num_on_track, action):
    if action == 'switch':
        return -num_on_track
    else:
        return -1

def max_expected_choise_worthiness(credence, num_on_track, boosted = False):
    actions = {'switch': 0, 'nothing': 0}

    penalty_deon_switch = penalty_deon(num_on_track, 'switch', boosted)
    penalty_deon_nothing = penalty_deon(num_on_track, 'nothing', boosted)

    penalty_util_switch = penalty_util(num_on_track, 'switch')
    penalty_util_nothing = penalty_util(num_on_track, 'nothing')

    actions['switch'] = credence * penalty_deon_switch + (1 - credence) * penalty_util_switch
    actions['nothing'] = credence * penalty_deon_nothing + (1 - credence) * penalty_util_nothing

    return max(actions, key=actions.get)

def classic_trolley_problem(credence_granularity, boosted = False):
    max_num_on_track = 10

    N = len(np.arange(0, 1.0, credence_granularity))

    output = np.zeros((len(np.arange(0, 1.0, credence_granularity)), len(np.arange(1, (max_num_on_track + 1)))))

    for index, credence in enumerate(np.linspace(0, 1.0, 11, endpoint=True)):
            for num_on_track in range(1, max_num_on_track):
                action = max_expected_choise_worthiness(credence, num_on_track, boosted)
                print(f"For credence {credence} in deontology and {num_on_track} people on the track, the best action is: {action}")

                if action == 'switch':
                    output[index][num_on_track] = 1
                else:
                    output[index][num_on_track] = 0

    return output

def mian():
    
    credence_granularity = 0.1 #granuality from 0 to 1, this is step
    
    output = classic_trolley_problem(credence_granularity)
    print(output)

    boosted = True

    output_boosted = classic_trolley_problem(credence_granularity, boosted)
    print(output_boosted)

if __name__ == "__main__":
    mian()