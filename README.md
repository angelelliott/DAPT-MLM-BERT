# DAPT-MLM-BERT
 
 
Based on <a href="https://arxiv.org/pdf/2004.10964.pdf">Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks</a> 
 
Domain-adaptive pre-training is the process of tailoring a pre-trained model to the domain of a target classification task. This is done by submitting the model to an additional round of pre-training on a large unlabeled domain-specific corpus. 
 

I submitted a pre-trained Spanish BERT model to an additional round of domain-adaptive masked language modeling pre-training for the task of misogynistic tweet detection in Spanish. 

Pre-training Corpus: "profanitiesdatasetprocessed.csv"
<ul>
  <li>5.5 M tweets in Spanish, each tweet contains at least one common Spanish profanity.</li>
  <li>Scraped using snscrape and Twitter API.</li>
  <li>Idea: 
   <ol>
   <li>BERT pre-trained on a general domain doesn't understand Twitter language.</li>
   <li>Misogynistic tweets usually have slurs.</li>
   </ol>
</ul>

