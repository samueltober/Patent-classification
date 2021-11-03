from absl import app
from absl import flags

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    print("The path to the checkpoint file", FLAGS.checkpoint_file)


if __name__ == '__main__':
    flags.DEFINE_string('checkpoint_file', None,
                        'The path to the checkpoint file')

    flags.mark_flag_as_required('checkpoint_file')

    app.run(main)
