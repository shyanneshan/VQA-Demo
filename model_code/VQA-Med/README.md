# JUST at VQA-Med: A VGG-Seq2Seq Model

http://ceur-ws.org/Vol-2125/paper_171.pdf

This paper describes the VGG-Seq2Seq system for the Medi- cal Domain Visual Question Answering (VQA-Med) Task of ImageCLEF 2018. The proposed system follows the encoder-decoder architecture, where the encoders fuses a pretrained VGG network with an LSTM net- work that has a pretrained word embedding layer to encode the input. To generate the output, another LSTM network is used for decoding. When used with a pretrained VGG network, the VGG-Seq2Seq model man- aged to achieve reasonable results with 0.06, 0.12, 0.03 BLEU, WBSS and CBSS, respectively. Moreover, the VGG-Seq2Seq is not expensive to train.
