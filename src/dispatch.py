
import argparse, logging, yaml
from loaders import *
from pipeline import *
import mlsim

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a config file and optionally specify a creator.")
    parser.add_argument("--action", choices=["train", 'sim.sample'], help="TBD")
    parser.add_argument("--model", type=str, default=None, help="TBD")
    parser.add_argument("--data", type=str, default=None, help="TBD")
    parser.add_argument("--cfg", type=str, default='configs/config.yaml', help="TBD")
    parser.add_argument("--models-cfg", dest='models_cfg', type=str, default='configs/models.yaml', help="TBD")
    parser.add_argument('--rewrite', action='store_true', help='rewrite if exists', default=False)
    parser.add_argument('--causal-tags', dest='causal_tags', default='MoviesCausalGPT')
    parser.add_argument("--nsamples", type=int, default=3, help="TBD")
    return parser.parse_args()

def validate_cfg(cfg, path=[]):
    missing = []
    for name, value in cfg.items():
        next_path = path + [name]
        if type(value) == dict:
            missing += validate_cfg(value, next_path)
        elif value is None:
            missing.append(".".join(next_path))
    if not path and missing:
        print("Missing config:")
        for x in missing:
            print(f' - {x}')
    return missing

def read_cfg(path):
    with open(path, "rt") as yfile:
        cfg = yaml.safe_load(yfile)
        validate_cfg(cfg)
        return cfg


        

if __name__ == '__main__':
        
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start")  

    args = parse_arguments()
    cfg = read_cfg(args.cfg)
    paths = PathProvider(cfg['paths']['results'], cfg['paths']['products'])
    
    if args.action == 'train':
        models_cfg = read_cfg(args.models_cfg)
        trainer = get_model_trainer(paths, models_cfg['models'], args.model)
        uidata = MovieLensData(get_uidata_loader(cfg, args.data))
        trainer.fit(uidata)

    if args.action == 'sim.sample':
        assert args.data == 'ml-1m'

        uidata = MovieLensData(get_uidata_loader(cfg, args.data))
        model = load_model(paths, uidata.name(), args.model)
        name = f'CausalSim.{uidata.name}.{args.model}'

        probs = model.probablity_matrix()
        causal_df = enrich_cause_indexes(
            pd.read_csv(paths.get_product_csv(args.causal_tags)), uidata.info)
        
        mlsim.create_sim_data_samples(paths, name, model, uidata, causal_df, 
                                      nsamples=args.nsamples, rewrite=args.rewrite)

    logging.info("done")    