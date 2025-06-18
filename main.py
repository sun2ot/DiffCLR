from utils import *
from utils.log import Log
from utils.conf import load_config, check_config
from train import Trainer
from data import DataHandler
from utils.loss import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Configs')
    parser.add_argument('--config', '-c', default='conf/test.toml', type=str, help='config file path')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        main_log = Log('main', config.data.name)
        config.base.timestamp = Log.log_time
        format_config(config, main_log)
        check_config(config)
        print(f"Load configuration ({config.data.name}) file successfully👌")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    set_seed(config.base.seed)

    data_handler = DataHandler(config)
    main_log.info('Load Data')
    data_handler.load_data()
    main_log.info(f"USER: {config.data.user_num}, ITEM: {config.data.item_num}")
    main_log.info(f"NUM OF INTERACTIONS: {len(data_handler.train_data)}")
    
    trainer = Trainer(data_handler, config, main_log)
    trainer.run()