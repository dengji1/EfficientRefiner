class Args:
    def __init__(self, args):
        self.gamma = args.gamma
        if not args.pruning:
            self.threads_per_block = 16
            self.blocks_per_grid = 16
            self.threads_per_block_2d = (16, 16)
            self.blocks_per_grid_2d = (16, 16)
        else:
            self.threads_per_block = 512
            self.blocks_per_grid = 2048
            self.threads_per_block_2d = (32, 32)
            self.blocks_per_grid_2d = (128, 128)
            self.percentile = 60
            self.min_bin_num = 512
            self.max_module_per_bin = 500
