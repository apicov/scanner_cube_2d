from gym.envs.registration import register

register(
    id='ScannerEnv-v2',
    entry_point='scan_gym_v2.envs.ScannerEnv:ScannerEnv',
)
