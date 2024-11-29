import os, sys
# enable ppocr module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from argsparser import ArgsParser, load_config, merge_config


def main(config):
    # parse configs
    # build dataloader
    # build model
    # training loop
    print(config)


if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profiler_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profiler_dic)
    main(config)