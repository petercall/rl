# Reward Model

In this repository I use full Supervised Fine Tuning (SFT) to fine-tune a model.

Note on the datasets used:
1. The original Bespoke dataset can be found at: bespokelabs/Bespoke-Stratos-17k. However, they use the "from" and "value" convention instead of "role" and "content". Because "role" and "content" form is much more popular, I use the dataset found at: HuggingFaceH4/Bespoke-Stratos-17k, which uses that convention, and has a "messages" column that can be directly used by TRL.
2. The open-thoughts dataset lacks a "messages" column, so I renamed the "conversations" column to "messages"