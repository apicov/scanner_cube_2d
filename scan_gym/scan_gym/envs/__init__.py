from gym.envs.registration import register

register(
    id='ScannerEnv-v1',
    entry_point='scan_gym.envs.ScannerEnv:ScannerEnv',
)

register(
    id='ScannerEnv-v2',
    entry_point='scan_gym.envs.ScannerEnv2:ScannerEnv',
)

register(
    id='ScannerEnv-v3',
    entry_point='scan_gym.envs.ScannerEnv3:ScannerEnv',
)

register(
    id='ScannerEnv-v4',
    entry_point='scan_gym.envs.ScannerEnv4:ScannerEnv',
)
