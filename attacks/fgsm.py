import tensorflow as tf


def create_adversarial_pattern(input_image, input_label, model):
    """
    Computes the signed gradient of the loss with respect to the input image.

    This is the core idea behind FGSM attack.
    """

    # Ensure model is in inference mode
    # (important if BatchNorm / Dropout exists)
    with tf.GradientTape() as tape:

        # Watch input image because normally TF does not compute
        # gradients for non-trainable tensors
        tape.watch(input_image)

        # Forward pass
        prediction = model(input_image, training=False)

        # Use appropriate loss
        loss = tf.keras.losses.BinaryCrossentropy()(input_label, prediction)

    # Compute gradient of loss w.r.t image
    gradient = tape.gradient(loss, input_image)

    # Take sign of gradient
    signed_grad = tf.sign(gradient)

    return signed_grad


def fgsm_attack(image, epsilon, model, label):
    """
    Performs Fast Gradient Sign Method attack.

    epsilon:
        Controls strength of attack.
        Small -> weak attack
        Large -> strong attack
    """

    # Generate adversarial perturbation
    perturbation = create_adversarial_pattern(image, label, model)

    # Add scaled noise to image
    adversarial_image = image + epsilon * perturbation

    # Clip values so pixels remain valid
    adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)

    return adversarial_image