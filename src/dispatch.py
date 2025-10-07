
import argparse, logging, yaml
from loaders import *
from pipeline import *
from utils import *
import mlsim

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a config file and optionally specify a creator.")
    parser.add_argument("--action", choices=["train", 'sim.sample', 'sim.gt'], help="TBD")
    parser.add_argument("--model", type=str, default=None, help="TBD")
    parser.add_argument("--data", type=str, default=None, help="TBD")
    parser.add_argument("--cfg", type=str, default='configs/config.yaml', help="TBD")
    parser.add_argument("--models-cfg", dest='models_cfg', type=str, default='configs/models.yaml', help="TBD")
    parser.add_argument('--rewrite', action='store_true', help='rewrite if exists', default=False)
    parser.add_argument('--causal-tags', dest='causal_tags', default='MoviesCausalGPT')
    parser.add_argument('--causal-enrich', dest='causal_enrich', default=1, type=int, help='1 to enrich causal indexes')
    parser.add_argument("--csim", type=str, default=None, help="TBD")
    parser.add_argument("--nsamples", type=int, default=3, help="TBD")
    parser.add_argument("--idx", type=int, default=0, help="TBD")
    parser.add_argument("--partition", type=int, default=None, help="TBD")
    return parser.parse_args()


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

    if args.action in ['sim.sample', 'sim.gt']:
        assert args.data == 'ml-1m'

        uidata = MovieLensData(get_uidata_loader(cfg, args.data))
        model = load_model(paths, uidata.name(), args.model)
        assert args.csim is not None
        name = f'{args.csim}.{uidata.name()}.{args.model}'

        causal_df = pd.read_csv(paths.get_product_csv(args.causal_tags))
        if args.causal_enrich:
            causal_df = enrich_cause_indexes(causal_df, uidata.info)
        
        if args.action == 'sim.sample':
            mlsim.create_sim_data_samples(paths, name, model, uidata, causal_df, 
                                        nsamples=args.nsamples, rewrite=args.rewrite)

        elif args.action == 'sim.gt':
            mlsim.create_ground_truth_samples(paths, name, model, uidata, causal_df, idx=args.idx, partition=args.partition)

    logging.info("done")    