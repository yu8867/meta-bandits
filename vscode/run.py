from main import Main
from policy.policy import RS, RS_OPT, RS_CH, SRS, SRS_OPT,SRS_CH, TS, UCB1T

"""メイン
Args:
    n_sims(int) : sim数
    steps(int) : step数
    k(int) : アーム数(行動数)
    unsteady : 定常0 非定常1
    policy : [下位エージェント, 上位エージェント]
"""

n_sim, steps, k, unsteady = 100, 10000, 20, 1
policy = [[SRS_OPT, SRS_CH]]

main = Main(policy, n_sim, steps, k, unsteady)
main.main()
    