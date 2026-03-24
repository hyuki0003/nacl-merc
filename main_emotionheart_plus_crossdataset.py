from data import get_MELD_loaders
import argparse

import os
import torch

import models

import utils
import graphdata as gdt

from datetime import datetime as dt

utils.make_route('./log', 'train.log')

log = utils.get_logger('./log/train.log')

def main(args1, args2):
    utils.set_seed(args1.seed)

    # load data
    log.info("Load pretraining dataset... Name: " + args1.dataset)
    pretraining_data_dir = os.path.join(os.getcwd(), args1.data_dir_path, args1.dataset)
    args1.data = os.path.join(pretraining_data_dir, "data_" + args1.dataset + ".pkl")

    log.info("Load finetuning dataset... Name: " + args2.dataset)
    finetuning_data_dir = os.path.join(os.getcwd(), args2.data_dir_path, args2.dataset)
    args2.data = os.path.join(finetuning_data_dir, "data_" + args2.dataset + ".pkl")

    if args1.dataset == "iemocap" or args1.dataset == "iemocap_4":
        dataset = utils.load_pkl(args1.data)
        graph_trainset_file = os.path.join(pretraining_data_dir, "graph_trainset.pkl")

        if not os.path.exists(graph_trainset_file):
            pretrainset = gdt.iemocap_4_graphDataset(dataset["train"], 'train', args1)
            utils.save_pkl(pretrainset, graph_trainset_file)
        pretrainset = utils.load_pkl(graph_trainset_file)

        # args.num_nodes = trainset.n_max_utterances * trainset.n_modalities  # with virtual node (graph token, e.g., CLS token in BERT)
        args1.n_max_utterances = pretrainset.n_max_utterances
        args1.n_max_speakers = pretrainset.n_max_speakers

        # if not os.path.exists(graph_devset_file):
        #     devset = gdt.iemocap_4_graphDataset(pretrainset["dev"], 'dev', args1)
        #     utils.save_pkl(devset, graph_devset_file)
        # devset = utils.load_pkl(graph_devset_file)
        #
        # if not os.path.exists(graph_testset_file):
        #     testset = gdt.iemocap_4_graphDataset(pretrainset["test"], 'test', args1)
        #     utils.save_pkl(testset, graph_testset_file)
        # testset = utils.load_pkl(graph_testset_file)

    elif args1.dataset == "meld":
        _, _, _, all_loader = get_MELD_loaders(args1.batch_size, args1.data)

        pretraining_graph_trainset_file = os.path.join(pretraining_data_dir, "graph_pretrainset.pkl")

        if not os.path.exists(pretraining_graph_trainset_file):
            pretrainset = gdt.meld_graphDataset(all_loader, 'pretrain', args1)
            utils.save_pkl(pretrainset, pretraining_graph_trainset_file)
        pretrainset = utils.load_pkl(pretraining_graph_trainset_file)
        args1.n_max_utterances = pretrainset.n_max_utterances
        args1.n_max_speakers = pretrainset.n_max_speakers

    elif args1.dataset == "mosei":
        all_loader = utils.load_pkl(args1.data)
        pretraining_graph_trainset_file = os.path.join(pretraining_data_dir, "graph_pretrainset.pkl")
        if not os.path.exists(pretraining_graph_trainset_file):
            pretrainset = gdt.mosei_graphDataset(all_loader, 'pretrain', args1)
            utils.save_pkl(pretrainset, pretraining_graph_trainset_file)
        pretrainset = utils.load_pkl(pretraining_graph_trainset_file)
        args1.n_max_utterances = pretrainset.n_max_utterances
        args1.n_max_speakers = pretrainset.n_max_speakers

    else:
        raise ValueError(f"Unknown pretraining dataset name: {args1.pretraining_dataset}")

    if args2.dataset == "iemocap_4" or args2.dataset == "iemocap":
        finetuning_data = utils.load_pkl(args2.data)
        graph_trainset_file = os.path.join(finetuning_data_dir, "graph_trainset.pkl")
        graph_devset_file = os.path.join(finetuning_data_dir, "graph_devset.pkl")
        graph_testset_file = os.path.join(finetuning_data_dir, "graph_testset.pkl")

        if not os.path.exists(graph_trainset_file):
            trainset = gdt.iemocap_4_graphDataset(finetuning_data["train"], 'train', args2)
            utils.save_pkl(trainset, graph_trainset_file)
        trainset = utils.load_pkl(graph_trainset_file)

        # args.num_nodes = trainset.n_max_utterances * trainset.n_modalities  # with virtual node (graph token, e.g., CLS token in BERT)
        args2.n_max_utterances = trainset.n_max_utterances
        args2.n_max_speakers = trainset.n_max_speakers

        if not os.path.exists(graph_devset_file):
            devset = gdt.iemocap_4_graphDataset(finetuning_data["dev"], 'dev', args2)
            utils.save_pkl(devset, graph_devset_file)
        devset = utils.load_pkl(graph_devset_file)

        if not os.path.exists(graph_testset_file):
            testset = gdt.iemocap_4_graphDataset(finetuning_data["test"], 'test', args2)
            utils.save_pkl(testset, graph_testset_file)
        testset = utils.load_pkl(graph_testset_file)

    elif args2.dataset == "meld":
        pass
    elif args2.dataset == "mosei":
        pass
    log.debug("Building emotionheart...")

    if args2.unimodal_inference and args2.modalities in ["a", "t", "v"]:
        model = torch.load(
            f"./{args1.save_model_checkpoint}/atv_best_model.pt",
            weights_only=False
        )

        print("--- Checking Model Parameters ---")
        # if args1.specific:
        #     for name, param in model.named_parameters():
        #         if not name.startswith("linear_fusion") and not name.startswith("classifier"):
        #             param.requires_grad=False
        #         print(f"'{name}'-{param.requires_grad}")
        #     print("--- Check Complete ---")
        # else:
        #     for name, param in model.named_parameters():
        #         if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith("classifier"):
        #             param.requires_grad=False
        #         print(f"'{name}'-{param.requires_grad}")
        #     print("--- Check Complete ---")
        for name, param in model.named_parameters():
            # if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith("classifier"):
            if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith(
                    "attention_fusion") and not name.startswith("classifier") and not name.startswith(
                    "unimodal_classifiers"):
                param.requires_grad = False
            print(f"'{name}'-{param.requires_grad}")
        print("--- Check Complete ---")

        model.args = args2
        model.modalities = args2.modalities
        model.n_modalities = len(args2.modalities)
        model.encoder.args = args2

        model.encoder.n_modalities = len(args2.modalities)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:     {total_params:,}")
    else:
        if args1.dataset == "iemocap":
            n_nodes = trainset.n_max_utterances
        else:
            n_nodes = max([trainset.n_max_utterances, testset.n_max_utterances, testset.n_max_utterances])
    encoder = models.EmotionHeartEncoder(args1, n_nodes)
    decoder = models.EmotionHeartDecoder(args1)
    model = models.EmotionHeartModel(args1, encoder, decoder).to(args1.device)


    opt1 = models.Optim(float(args1.learning_rate), int(args1.T), float(args1.max_grad_value), float(args1.weight_decay),
                        int(args1.epochs),len(pretrainset))
    opt1.set_parameters(model.parameters(), args1.optimizer)
    sched1 = opt1.get_scheduler(args1.scheduler)

    print(f"args_pretrain:\n{args1}\n")
    print(f"args_finetune:\n{args2}")
    coach = models.Coach(pretrainset, trainset, devset, testset, model, opt1, sched1, args1, args2, log)
    # if not args.from_begin:
    #     ckpt = torch.load(model_file)
    #     coach.load_ckpt(ckpt)
    #     print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()
    # Save.
    metrics = {
        "best_dev_f1": ret[0],
        "best_dev_acc": ret[1],
        "best_epoch": ret[2],
        "best_state": ret[3],
        "train_losses": ret[4],
        "dev_losses": ret[5],
        "dev_f1s": ret[6],
        "test_f1s": ret[7],
        "dev_accs": ret[8],
        "test_accs": ret[9],
        "test_losses": ret[10],
    }
    save_loss_plot_path = os.path.join(os.getcwd(), args1.save_analysis_path+'_'+args2.dataset,
                                       "loss_plot_"+ dt.now().strftime('%Y-%m-%d-%H-%M-%S')+".png")

    utils.plot_and_save_loss(ret[4], ret[5], ret[10], filename=save_loss_plot_path)

    save_metrics_path = os.path.join(os.getcwd(), args1.save_analysis_path + '_' + args2.dataset,
                                          "metrics_" + dt.now().strftime('%Y-%m-%d-%H-%M-%S') + ".log")
    os.makedirs(os.path.dirname(save_metrics_path), exist_ok=True)

    with open(save_metrics_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":

    #dataset list: ["iemocap", "iemocap_4", "mosei", "meld"]
    dataset1 = "meld"
    dataset2 = "iemocap"
    
    parser1 = argparse.ArgumentParser(description="pretraining_data")

    parser1.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default=dataset1,
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name."
    )
    # parser.add_argument(
    #     "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    # )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in UnimodalEncoder.py"""
    # parser1.add_argument(
    #     "--modalities",
    #     type=str,
    #     default="atv",
    #     # required=True,
    #     choices=["a", "t", "v", "at", "tv", "av", "atv"],
    #     help="Modalities",
    # )

    parser1.add_argument(
        "--relation_type",
        default="eam",
        choices=["e", "ea", "eam"],
        help="Choose relation contruct type. e: interlocuter, a: intralocuter, m: intermodality",
    )

    parser1.add_argument(
        "--optimizer",
        default="adamw",
        choices=["sgd","rmsprop","adam","adamw"],
        help="Choose optimizer",
    )

    temp = parser1.parse_args()

    config_path = "config/"
    args1 = utils.get_config_args(parser1, config_path+temp.dataset+'_pretrain.yaml', dataset=temp.dataset)

    args1.num_edges = len(args1.relation_type)
    args1.num_degree = len(args1.relation_type)
    if args1.relation_type == "eam" and args1.modalities == "atv":
        args1.num_degree += 1

    args1.ffn_embed_dim = args1.encoder_embed_dim * args1.ffn_embed_scaler

    args1.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
        },
        "meld": {
            "a":300,
            "t":600,
            "v":342
        }
    }

    # parser.add_argument(
    #     "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    # )
    parser2 = argparse.ArgumentParser(description="finetuning_data")
    parser2.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default=dataset2,
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name."
    )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in UnimodalEncoder.py"""
    # parser2.add_argument(
    #     "--modalities",
    #     type=str,
    #     default="atv",
    #     # required=True,
    #     choices=["a", "t", "v", "at", "tv", "av", "atv"],
    #     help="Modalities",
    # )

    parser2.add_argument(
        "--relation_type",
        default="eam",
        choices=["e", "ea", "eam"],
        help="Choose relation contruct type. e: interlocuter, a: intralocuter, m: intermodality",
    )

    parser2.add_argument(
        "--optimizer",
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Choose optimizer",
    )

    temp2 = parser2.parse_args()

    args2 = utils.get_config_args(parser2, config_path+temp2.dataset+'.yaml', dataset=temp2.dataset)

    args2.num_edges = len(args2.relation_type)
    args2.num_degree = len(args2.relation_type)
    if args2.relation_type == "eam" and args2.modalities == "atv":
        args2.num_degree += 1

    args2.ffn_embed_dim = args2.encoder_embed_dim * args2.ffn_embed_scaler

    args2.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
        },
        "meld": {
            "a": 300,
            "t": 600,
            "v": 342
        }
    }


    log.debug(args1)
    log.debug(args2)

    main(args1, args2)
