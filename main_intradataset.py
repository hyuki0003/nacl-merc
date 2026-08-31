"""NACL-MERC intra-dataset training (train-from-scratch or pretrain+finetune on one dataset).

Examples:
    # MELD from scratch (supervised + NACL regularizer, per config/meld.yaml)
    python main_intradataset.py --dataset meld

    # IEMOCAP 6-way intra-dataset
    python main_intradataset.py --dataset iemocap

    # Evaluate the saved fine-tuned checkpoint on the test set only
    python main_intradataset.py --dataset meld --eval_only

Stage behavior is controlled by config/<dataset>_pretrain.yaml (stage 1) and
config/<dataset>.yaml (stage 2), as in main_crossdataset.py.
"""

import argparse
import os
from datetime import datetime as dt

import torch

import graphdata as gdt
import models
import utils
from data import get_MELD_loaders

utils.make_route('./log', 'train.log')
log = utils.get_logger('./log/train.log')


def load_data(args):
    """Build (or load the cached) train/dev/test graph datasets for one dataset."""
    data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(data_dir, "data_" + args.dataset + ".pkl")

    graph_trainset_file = os.path.join(data_dir, "graph_trainset.pkl")
    graph_devset_file = os.path.join(data_dir, "graph_devset.pkl")
    graph_testset_file = os.path.join(data_dir, "graph_testset.pkl")

    if args.dataset in ("iemocap", "iemocap_4"):
        data = None
        sets = {}
        for split, graph_file in (("train", graph_trainset_file),
                                  ("dev", graph_devset_file),
                                  ("test", graph_testset_file)):
            if not os.path.exists(graph_file):
                if data is None:
                    data = utils.load_pkl(args.data)
                sets[split] = gdt.iemocap_4_graphDataset(data[split], split, args)
                utils.save_pkl(sets[split], graph_file)
            else:
                sets[split] = utils.load_pkl(graph_file)
        trainset, devset, testset = sets["train"], sets["dev"], sets["test"]

    elif args.dataset == "meld":
        if not os.path.exists(graph_trainset_file) or not os.path.exists(graph_testset_file):
            train_loader, _, test_loader, _ = get_MELD_loaders(args.batch_size, args.data)

        if not os.path.exists(graph_trainset_file):
            trainset = gdt.meld_graphDataset(train_loader, 'train', args)
            utils.save_pkl(trainset, graph_trainset_file)
        else:
            trainset = utils.load_pkl(graph_trainset_file)

        if not os.path.exists(graph_testset_file):
            testset = gdt.meld_graphDataset(test_loader, 'test', args)
            utils.save_pkl(testset, graph_testset_file)
        else:
            testset = utils.load_pkl(graph_testset_file)

        # MELD ships no separate dev split in this setup; dev == test.
        devset = testset

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    args.n_max_utterances = trainset.n_max_utterances
    args.n_max_speakers = trainset.n_max_speakers
    return trainset, devset, testset


def main(args1, args2):
    utils.set_seed(args1.seed)

    log.info("Load dataset... Name: " + args1.dataset)
    trainset, devset, testset = load_data(args1)
    args2.n_max_utterances = args1.n_max_utterances
    args2.n_max_speakers = args1.n_max_speakers

    # Apply the missing-modality override only AFTER the graph datasets are built
    # (see main_crossdataset.py).
    override = getattr(args2, "inference_modalities", None)
    if override:
        args2.modalities = override
        args2.num_degree = len(args2.relation_type)
        if args2.relation_type == "eam" and args2.modalities == "atv":
            args2.num_degree += 1

    log.debug("Building model...")
    finetuned_ckpt = f"./{args1.save_model_checkpoint}_{args2.dataset}/finetune_{args1.modalities}_best_model.pt"

    load_finetuned_for_inference = (
        not args2.do_finetune and not args2.from_scratch and args2.unimodal_inference
    )
    if args2.eval_only or load_finetuned_for_inference:
        # Inference-only: no pretraining stage.
        args1.from_begin = False

    if args2.eval_only:
        # Coach loads the fine-tuned checkpoint and evaluates the test set.
        model = None
    elif load_finetuned_for_inference:
        # Missing-modality inference: reuse the fine-tuned model with a reduced modality set.
        model = torch.load(finetuned_ckpt, map_location=args2.device, weights_only=False)
        for name, param in model.named_parameters():
            if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith("classifier"):
                param.requires_grad = False
        model.args = args2

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:     {total_params:,}")
    else:
        if args1.dataset == "iemocap":
            n_nodes = trainset.n_max_utterances
        else:
            n_nodes = max(trainset.n_max_utterances, testset.n_max_utterances)
        encoder = models.EmotionHeartEncoder(args1, n_nodes)
        decoder = models.EmotionHeartDecoder(args1)
        model = models.EmotionHeartModel(args1, encoder, decoder).to(args1.device)

    opt1, sched1 = None, None
    if model is not None:
        opt1 = models.Optim(float(args1.learning_rate), int(args1.T), float(args1.max_grad_value),
                            float(args1.weight_decay), int(args1.epochs), len(trainset))
        opt1.set_parameters(model.parameters(), args1.optimizer)
        sched1 = opt1.get_scheduler(args1.scheduler)

    print(f"args_pretrain:\n{args1}\n")
    print(f"args_finetune:\n{args2}")
    coach = models.Coach(trainset, trainset, devset, testset, model, opt1, sched1, args1, args2, log)

    log.info("Start training...")
    ret = coach.train()

    if args2.do_finetune:
        metrics = {
            "best_dev_f1": ret[0],
            "best_dev_acc": ret[1],
            "best_epoch": ret[2],
            "train_losses": ret[4],
            "dev_losses": ret[5],
            "dev_f1s": ret[6],
            "test_f1s": ret[7],
            "dev_accs": ret[8],
            "test_accs": ret[9],
            "test_losses": ret[10],
            "best_test_f1": ret[11],
        }
        analysis_dir = os.path.join(os.getcwd(), args1.save_analysis_path + '_' + args2.dataset)
        timestamp = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

        utils.plot_and_save_loss(ret[4], ret[5], ret[10],
                                 filename=os.path.join(analysis_dir, f"loss_plot_{timestamp}.png"))

        save_metrics_path = os.path.join(analysis_dir, f"metrics_{timestamp}.log")
        os.makedirs(os.path.dirname(save_metrics_path), exist_ok=True)
        with open(save_metrics_path, "w", encoding="utf-8") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NACL-MERC intra-dataset training")
    parser.add_argument(
        "--dataset", type=str, default="meld",
        choices=["meld", "iemocap"],
        help="Dataset (uses config/<name>_pretrain.yaml and config/<name>.yaml)."
    )
    parser.add_argument("--config_dir", type=str, default="config", help="Directory containing YAML configs.")
    parser.add_argument("--pretrain_config", type=str, default=None,
                        help="Explicit stage-1 YAML path (default: <config_dir>/<dataset>_pretrain.yaml). "
                             "The encoder architecture is taken from this config even for from-scratch runs; "
                             "use config/meld_pretrain_scratch.yaml for the paper's MELD from-scratch setting.")
    parser.add_argument("--finetune_config", type=str, default=None,
                        help="Explicit stage-2 YAML path (default: <config_dir>/<dataset>.yaml).")
    parser.add_argument(
        "--relation_type", type=str, default="eam", choices=["e", "ea", "eam"],
        help="Graph relations. e: interlocutor, a: intralocutor, m: intermodality."
    )
    parser.add_argument("--pretrain_optimizer", type=str, default="adamw",
                        choices=["sgd", "rmsprop", "adam", "adamw"])
    parser.add_argument("--finetune_optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam", "adamw"])
    parser.add_argument("--device", type=str, default=None,
                        help="Device override for both stages (e.g. cuda:1).")
    parser.add_argument("--modalities", type=str, default=None,
                        choices=["a", "t", "v", "at", "tv", "av", "atv"],
                        help="Fine-tuning/inference modality override (used for missing-modality inference).")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and evaluate the saved fine-tuned checkpoint on the test set.")
    cli = parser.parse_args()

    pretrain_cfg = cli.pretrain_config or os.path.join(cli.config_dir, cli.dataset + "_pretrain.yaml")
    finetune_cfg = cli.finetune_config or os.path.join(cli.config_dir, cli.dataset + ".yaml")
    args1 = utils.load_config(pretrain_cfg, cli.dataset, cli.relation_type, cli.pretrain_optimizer, cli.device)
    args2 = utils.load_config(finetune_cfg, cli.dataset, cli.relation_type, cli.finetune_optimizer, cli.device)

    if cli.modalities is not None:
        if args2.do_finetune or not args2.unimodal_inference:
            parser.error(
                "--modalities is only for missing-modality inference: set "
                "do_finetune: false and unimodal_inference: true in the target config."
            )
        # Applied in main() after the graph datasets are built.
        args2.inference_modalities = cli.modalities

    args2.eval_only = cli.eval_only
    if cli.eval_only:
        args1.from_begin = False
        args2.do_finetune = False
        args2.from_scratch = False
        args2.unimodal_inference = False

    log.debug(args1)
    log.debug(args2)

    main(args1, args2)
