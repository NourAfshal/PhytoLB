# EmoFlow
1 Introduction
When people talk to each other, their emotions are
always changing. Someone might start a conver-
sation feeling calm, then get excited, then frus-
trated, and finally feel relieved once things are
sorted out. This is just how human communication
works. However, most emotion detection systems
today look at each message on its own and try to
label it with one emotion. They do not take into
account what was said before or how the conver-
sation got to that point. We think this is a big gap,
and our project is built around fixing it.
Think about a simple example on Reddit. A per-
son asks a question and they are curious. Someone
gives a great answer and they become happy. Then
another person jumps in with a rude comment and
the mood shifts to anger. By the end, the original
person might feel relief once everything is cleared
up. If you only look at one message at a time, you
miss the whole picture. But if you look at the full
conversation, the emotional story becomes clear.
Our project, which we call EmoFlow, is de-
signed to track and predict how emotions change
throughout a conversation. Instead of just asking
"what emotion is this message?", we ask "what
emotion is likely to come next, given everything
that has been said so far?" This makes the problem
much more interesting and useful in real life.
Dataset. We will be using GoEmotions (Dem-
szky et al., 2020), a dataset made by Google Re-
search. It contains 58,000 Reddit comments that
were labeled by real people with 27 different
emotion categories, such as admiration, curiosity,
grief, excitement, and nervousness, along with a
neutral label. Most emotion datasets only use 6
or 7 basic emotions, so having 27 gives us a much
more detailed and realistic picture of how people
actually feel. Another great thing about this dataset
is that Reddit comments are part of conversation
threads, which means we can rebuild full conver-
sations and study how emotions change from one
message to the next.
Approach. We plan to work in three steps. First,
we will rebuild conversation threads from the
dataset so we can see the full flow of a discus-
sion. Second, we will train models to learn how
emotions tend to move from one turn to the next.
We will start with a simple Markov chain model
to understand the basic patterns, and then move to
an LSTM neural network that can capture longer
context. Third, we will build emotion flow visual-
izations, which are timeline charts that show how
the mood of a conversation shifts from start to fin-
ish.
Why It Matters. This kind of system has many
real-world uses. It could help build chatbots that ac-
tually understand how a user is feeling and respond
in a more caring way. It could also be used to spot
when a conversation is heading toward conflict or
when someone might be going through a hard time
emotionally. Customer service platforms could use
it to detect frustration before it gets worse, and
researchers could use it as a tool to study mental
health patterns in online communities.
References
Dorottya Demszky, Dana Movshovitz-Attias, Jeong-
wook Ko, Alan Cowen, Gaurav Nemade, and Sujith
Ravi. 2020. Goemotions: A dataset of fine-grained
emotions. In Proceedings of the 58th Annual Meet-
ing of the Association for Computational Linguistics,
pages 4040–4054.
1
