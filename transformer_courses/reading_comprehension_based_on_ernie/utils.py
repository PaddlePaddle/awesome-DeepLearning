import paddle

class CrossEntropyLossForRobust(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForRobust, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y   # both shape are [batch_size, seq_len]
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=start_logits, label=start_position, soft_label=False)
        start_loss = paddle.mean(start_loss)
        end_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=end_logits, label=end_position, soft_label=False)
        end_loss = paddle.mean(end_loss)

        loss = (start_loss + end_loss) / 2
        return loss



