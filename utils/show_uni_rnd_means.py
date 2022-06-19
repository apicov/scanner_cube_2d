import json

with open('uni_rnd_policy_runs_05_v3.json') as f:
    data = json.load(f)
 
for p in sorted(data.keys()):
    print("p {} r_rew_mean {:.4f} r_rew_std {:.4f} r_gt_mean {:.4f} r_gt_std {:.4f} u_mean {:.4f} u_std {:.4f}".format(p,data[p]['agent_rnd']['cum_reward_mean'], data[p]['agent_rnd']['cum_reward_std'],
        data[p]['agent_rnd']['gt_mean'],data[p]['agent_rnd']['gt_std'],data[p]['uni']['gt_mean'],data[p]['uni']['gt_std'] ) )
