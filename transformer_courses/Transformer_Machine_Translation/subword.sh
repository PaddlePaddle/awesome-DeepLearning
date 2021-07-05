subword-nmt learn-bpe -s 32000 < zh-en/train.tags.zh-en.zh.cut.txt > zh-en/bpe.ch.32000

subword-nmt learn-bpe -s 32000 < zh-en/train.tags.zh-en.en.txt > zh-en/bpe.en.32000

subword-nmt apply-bpe -c zh-en/bpe.ch.32000 < zh-en/train.tags.zh-en.zh.cut.txt > zh-en/train.ch.bpe
subword-nmt apply-bpe -c zh-en/bpe.ch.32000 < dev_cn.cut.txt > zh-en/dev.ch.bpe
subword-nmt apply-bpe -c zh-en/bpe.ch.32000 < test_cn.cut.txt > zh-en/test.ch.bpe
subword-nmt apply-bpe -c zh-en/bpe.en.32000 < zh-en/train.tags.zh-en.en.txt > zh-en/train.en.bpe
subword-nmt apply-bpe -c zh-en/bpe.en.32000 < dev_en.txt > zh-en/dev.en.bpe
subword-nmt apply-bpe -c zh-en/bpe.en.32000 < test_en.txt > zh-en/test.en.bpe

subword-nmt  get-vocab -i zh-en/train.ch.bpe -o zh-en/temp1

subword-nmt  get-vocab -i zh-en/train.en.bpe -o zh-en/temp2