# Out of domain detection via text noise

## Quickstart

```
python3 main.py -c configs/CONFIG.yml
```


## Log

Initial idea is to use a language model and a bayesian classifier outputing normalized outputs on a, hypershere.
We then train the classifier to output the same anchor X on the same point on the sphere and a gradualy noisier input with a higher cossine similarity.
At inference time, the predictive entropy will then me a good representation of the ID/OOD nature of the sample.

```
     ┌───────────────┐
     │               │   ┌───────────────────┐
     │Language      ─┼──►│ Projector (Baye)  ├───►
────►│Model          │   │                   │   On hypershere
     │               │   └───────────────────┘   of choosen size
     └───────────────┘
```

TODO: Insert sphere figure

---

Problem seems to arise from the function mapping the noise level from text and cosine dissimilarity, resulting in failed training

![](./research_logs/loss_fail1.png)

From top to bottom we try different loss scalers and schedulers for the strength of the the noise.
The last loss is when we disable the noise alltogether. Note that this does not mean the model is learning, as we did not account for training collapse.

---

Language models are already capable of making distinction between datasets, further training lowers the OOD discrimination capability.

---

Sometimes the OOD detection is not evident and further training might be required. In these cases, CIDER can be tried.
The situations apparently arrise when datasets use the sort of etxt semantic (e.g tweets) while the content is different in nature.
