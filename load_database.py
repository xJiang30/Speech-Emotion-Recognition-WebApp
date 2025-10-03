import config
from data_utils import parse_crema, parse_ravdess, parse_savee, parse_tess
from plot_utils import plot_emotion_distribution
import warnings
warnings.filterwarnings("ignore")

#load config
config = config.load_config("config.yaml")

# load and save to CSV
crema_df = parse_crema(config['data_paths']['Crema'])
plot_emotion_distribution(crema_df, "Crema")

ravdess_df = parse_ravdess(config['data_paths']['Ravdess'])
plot_emotion_distribution(ravdess_df, "Ravdess")

savee_df = parse_savee(config['data_paths']['Savee'])
plot_emotion_distribution(savee_df, "Savee")

tess_df = parse_tess(config['data_paths']['Tess'])
plot_emotion_distribution(tess_df, "Tess")
