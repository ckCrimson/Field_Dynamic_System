from Research_paper_systems_implementation.dice_system_implementation.state.dice_multi_step_reaching import \
    DiceMultiStepReaching
from Research_paper_systems_implementation.dice_system_implementation.state.dice_reachable import DiceReachable
from Research_paper_systems_implementation.dice_system_implementation.state.dice_reaching import DiceReaching
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace

# ---------------Constants --------------------#
ALPHA = 2
N_0=10
D=6
# --------- State ---------------#

mu_0 = DiceState(1.2)
print("Initial State: ",mu_0)

#-------------- State Space --------------#

dice_space = DiceStateSpace(mu_0,D,1,6)
print("Dice Space",dice_space)

# -----------Reachable --------------------#

dice_reachable = DiceReachable(ALPHA,N_0,D)
reachable_space = dice_reachable.get_reachable(mu_0)
print("reachable Space: ",dice_reachable.get_reachable(mu_0))

#--------------- Union Intersection -------------#

union_space = reachable_space.union_state_space(dice_space)
print("Union Space: ",union_space)
intersection_space = reachable_space.intersection_state_space(dice_space)
print("Intersection Space: ",intersection_space)

# -------------- Dice Reaching --------------------#

dice_reaching  =  DiceReaching(ALPHA,N_0,D)
reaching_space = dice_reaching.get_reaching(mu_0)
print("Reaching Space: ",dice_reaching.get_reaching(mu_0))

# ------------- Dice MultiStep reaching -----------------#

dice_multi_step_reaching = DiceMultiStepReaching(ALPHA,N_0,D)
multi_step_reaching_space = dice_multi_step_reaching.get_multi_step_reaching(mu_0,1)
print("Multi Step Reaching space",multi_step_reaching_space)

