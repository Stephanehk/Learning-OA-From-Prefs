import math
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


from learn_advantage.algorithms.models import RewardFunctionRegret, RewardFunctionPR
from learn_advantage.utils.pref_dataset_utils import CustomDataset
from learn_advantage.utils.segment_feats_utils import format_X, format_y


def get_reward_vec(args, model):
    for param in model.parameters():
        reward_vector = param.detach().cpu().numpy()
    if len(reward_vector) == 1:
        reward_vector = reward_vector[0]

    if args.use_extended_SF and args.generalize_SF:
        # Not implemented yet
        assert False
    return reward_vector


def reward_pred_loss(output, target):
    """
    Calculates cross entropy loss between predicted and ground truth preferences
    """
    output = torch.squeeze(output)
    output = torch.clamp(output, min=1e-35, max=None)
    output = torch.log(output)
    res = torch.mul(output, target)
    return -torch.sum(res)


def run_single_set(model, optimizer, X_train, y_train):
    """
    Runs a single training iteration for the given preference model on a batch of data
    """
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(X_train)
    y_pred = torch.clamp(y_pred, min=1e-35, max=None)  # prevents prob pred of 0

    loss = reward_pred_loss(y_pred, y_train)
    batch_size = y_pred.size()[0]

    loss /= batch_size

    return loss, optimizer, model


def train(
    *,
    aX,
    ay,
    args,
    plot_loss=True,
    preference_weights=None,
    env=None,
    validation_X=None,
    validation_y=None,
    check_point=False,
    shuffle_train_data=True,
):
    """
    Trains the preference model

    Input:
        - aX: a list of each preference pairs features
        - ay: a list of preferences for each segment pair
        - loss_coef: optional coefficient for loss function (default is 1)
        - plot_loss: plot loss after training
        - preference_weights: optional weights for deterministic regret model
        - gt_rew_vec: the ground truth reward vector, None if using the default delivery domain
        - env: the environment, None if using the default delivery domain
        - validation_X: a list of each preference pairs features that are to be used for validation
        - validation_y: a list of preferences for each segment pair that are to be used for validation
        - check_point: If true, then the models weights are saved at each epoch in the check_points array (defined locally in function)
    """

    # These checkpoints are the # of epochs at which we will save the current learned parameters
    check_points = [1, 10, 100, 200, 500, 1000, 10000, 30000]

    # format the preference dataset
    torch.manual_seed(0)  # for exact reproducibility
    X_train = format_X(aX)
    if hasattr(ay[0], "__len__"):
        y_train = format_y(ay, "arr")
    else:
        y_train = format_y(ay, "scalar")

    X_train = X_train.to(args.device)
    y_train = y_train.to(args.device)

    if validation_X is not None:
        validation_X = format_X(validation_X)
        validation_y = format_y(validation_y, "arr")
        validation_X = validation_X.to(args.device)
        validation_y = validation_y.to(args.device)

    # Compute the number of parameters we are learning. If we are learning an optimal advantage function, the number of parameters is |S| x |A|.
    if args.use_extended_SF:
        if env is not None:
            n_feats = 4 * env.width * env.height
        else:
            n_feats = 400
    else:
        n_feats = 6

    if args.preference_assum == "pr":
        # Learning with the partial return assumption
        model = RewardFunctionPR(args.gamma, n_features=n_feats)
    elif args.preference_assum == "regret":
        # Learning with the regret assumption
        model = RewardFunctionRegret(
            args,
            preference_weights,
            n_features=n_feats,
        )

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    losses = []
    val_losses = []

    for param in model.parameters():
        best_weights = param.detach().cpu().numpy()[0]

    if args.use_extended_SF and args.generalize_SF:
        batch_size = 4096
        n_batches = math.ceil(X_train.size(0) / batch_size)
    elif args.use_extended_SF and args.preference_assum == "regret":
        batch_size = 512
        n_batches = math.ceil(X_train.size(0) / batch_size)
    else:
        batch_size = X_train.size(0)
        n_batches = 1

    dataset = CustomDataset(X_train, y_train)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train_data
    )

    if validation_X is not None:
        validation_dataset = CustomDataset(validation_X, validation_y)
        validation_dataloader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True
        )
        n_validation_batches = math.ceil(validation_X.size(0) / batch_size)

    if args.device.type != "cpu":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    saved_models = []
    training_times = []
    best_total_loss = float("inf")
    for epoch in range(args.N_ITERS):
        batch_loss = 0
        val_batch_loss = 0
        for _ in range(n_batches):
            X_train_batch, y_train_batch = next(iter(train_dataloader))
            loss, optimizer, model = run_single_set(
                model, optimizer, X_train_batch, y_train_batch
            )
            detached_loss = loss.detach().cpu().numpy()
            if detached_loss < best_total_loss:
                best_total_loss = detached_loss
                for param in model.parameters():
                    best_weights = param.detach().cpu().numpy()[0].copy()

            loss.backward()
            optimizer.step()
            batch_loss += detached_loss

        if validation_X is not None:
            for _ in range(n_validation_batches):
                with torch.no_grad():
                    validation_X_batch, validation_y_batch = next(
                        iter(validation_dataloader)
                    )
                    val_loss, _, _ = run_single_set(
                        model,
                        optimizer,
                        validation_X_batch,
                        validation_y_batch,
                    )
                    val_batch_loss += val_loss.detach().cpu().numpy()

        losses.append(batch_loss / n_batches)
        if validation_X is not None:
            val_losses.append(val_batch_loss / n_validation_batches)

        if check_point and epoch + 1 in check_points:
            if args.device.type != "cpu":
                end_event.record()
                torch.cuda.synchronize()
                if len(training_times) > 0:
                    training_times.append(
                        training_times[-1] + start_event.elapsed_time(end_event)
                    )
                else:
                    training_times.append(start_event.elapsed_time(end_event))

                del end_event
                del start_event

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            saved_models.append(get_reward_vec(args, model))

    if plot_loss:
        plt.figure()
        plt.plot(losses, color="b")
        plt.plot(val_losses, color="r")

        header_parts = [
            f"{args.extra_details}",
            f"{args.learn_oaf}",
            f"{args.preference_model}",
            f"{args.preference_assum}",
            f"mode={args.mode}",
            f"extended_SF={args.use_extended_SF}",
            f"generalize_SF={args.generalize_SF}",
            f"num_prefs={len(X_train)}",
        ]
        header = "_".join(header_parts)
        if not os.path.exists("data/results/loss_graphs/"):
            os.makedirs("data/results/loss_graphs/")
        file_name = f"data/results/loss_graphs/{header}_loss_{args.LR}_{env}.png"
        plt.savefig(file_name)

    if args.preference_assum == "regret":
        for param in model.parameters():
            param.data = nn.parameter.Parameter(
                torch.tensor(best_weights).double().to(args.device)
            )

    for param in model.parameters():
        reward_vector = param.detach().cpu().numpy()
    if len(reward_vector) == 1:
        reward_vector = reward_vector[0]

    if args.use_extended_SF and args.generalize_SF:
        # we want to use the neural network as our reward vector
        reward_vector = model
    else:
        del model

    if check_point:
        return saved_models, losses, losses[-1], training_times
    return reward_vector, losses, losses[-1], training_times
