---
title: "Comparaison de pipelines de déploiement d'un modèle d'IA sur CPU et GPU"
author: "Pierre-Claver Olivier KOFFI"
date: "Novembre 2026"
lang: fr
---

# 1. Introduction

Ce TP vise à comprendre les couches d'une chaîne de déploiement d'un modèle d'intelligence artificielle et à mesurer leur effet sur le débit, la latence, la précision et la consommation énergétique. À partir d'un même modèle ResNet-18 écrit en PyTorch, quatre pipelines ont été mis en place, deux sur CPU et deux sur GPU, puis comparés selon un protocole identique.

# 2. Pipelines étudiés

## 2.1 Couches logicielles

Une chaîne de déploiement se décompose en cinq couches : le haut niveau (le framework d'entraînement, ici PyTorch), le format intermédiaire du modèle (ONNX ou un graphe PyTorch compilé), le middle level (le runtime ou le compilateur qui optimise le graphe : fusion d'opérateurs, choix des kernels), le bas niveau (les bibliothèques de calcul : oneDNN sur CPU, CUDA et cuDNN sur GPU) et enfin le matériel.

## 2.2 Pipeline CPU 1 : PyTorch, ONNX, OpenVINO

Le modèle est exporté au format ONNX (opset 17), puis compilé par OpenVINO Runtime pour le CPU avec l'indice de performance `LATENCY`. La compilation prend environ 0,29 s et n'est pas incluse dans les mesures.

## 2.3 Pipeline CPU 2 : PyTorch, torch.compile

Le modèle PyTorch est compilé avec le backend Inductor de `torch.compile`. La première inférence, qui déclenche la compilation, dure environ 33 s et est exclue des mesures. Une variante TorchScript (`torch.jit.trace` puis `torch.jit.freeze`) a été ajoutée pour comparer deux mécanismes natifs de PyTorch.

## 2.4 Pipeline GPU 1 : PyTorch, ONNX, TensorRT

Le modèle est exporté en ONNX puis converti en moteur TensorRT (`build_serialized_network`). Le moteur FP32 a été construit en environ 8 s. La tentative de build FP16 a échoué : l'API de l'installation TensorRT disponible sur Colab n'expose plus le drapeau `BuilderFlag.FP16` utilisé dans le code. Le pipeline est donc évalué uniquement en FP32 (voir la section 5.3).

## 2.5 Pipeline GPU 2 : PyTorch, torch.compile, CUDA

Même approche que le pipeline CPU 2, sur GPU T4 : le modèle est déplacé sur CUDA puis compilé par Inductor. La première inférence dure environ 22 s (compilation incluse), exclue des mesures.

# 3. Protocole expérimental

- **Modèle :** ResNet-18 initialisé sur ImageNet, dernière couche remplacée par une couche à 10 sorties, puis fine-tuné 2 époques sur 10 000 images de CIFAR-10 (optimiseur SGD, learning rate 0,01, précision mixte).
- **Données :** 50 images tirées au hasard dans le jeu de test CIFAR-10 (graine 42), identiques pour les quatre pipelines.
- **Prétraitement :** redimensionnement de 32x32 à 224x224, puis normalisation ImageNet (moyenne `[0.485, 0.456, 0.406]`, écart-type `[0.229, 0.224, 0.225]`).
- **Batch :** 10 images pour tous les pipelines.
- **Répétitions :** 3 passes de chauffe, puis 5 runs indépendants de 5 passes chacun, soit 25 passes sur 50 images par pipeline.
- **Indicateurs :** débit (images/s), temps total pour 50 images, latence par batch (ms), Top-1, énergie par image (J/image). Pour chaque indicateur : moyenne, écart-type, minimum et maximum sur les runs.
- **Énergie :** sur GPU, la puissance est lue via NVML pendant 10 s de traitement en continu, puis l'énergie par image est déduite. Sur CPU, le compteur RAPL n'est pas accessible sur Colab : l'énergie CPU n'est pas mesurée.
- **Matériel :** GPU NVIDIA T4 (Google Colab), CPU de la machine Colab, PyTorch récent avec Inductor.

# 4. Résultats

## 4.1 Tableau récapitulatif

Valeurs moyennes ± écart-type sur 5 runs, avec minimum et maximum entre crochets.

| Pipeline | Débit (img/s) | Temps 50 images (s) | Latence batch (ms) | Top-1 | Énergie (J/img) |
|---|---|---|---|---|---|
| CPU 1 : ONNX + OpenVINO | 14,06 ± 0,62 [13,25 ; 14,66] | 3,561 ± 0,159 | 712,1 ± 31,9 [682,2 ; 754,9] | 0,92 | non mesurée |
| CPU 2 : torch.compile | 13,98 ± 0,23 [13,71 ; 14,18] | 3,578 ± 0,059 | 715,6 ± 11,7 [705,2 ; 729,4] | 0,92 | non mesurée |
| CPU 2 bis : TorchScript | _à compléter_ | _à compléter_ | _à compléter_ | _à compléter_ | non mesurée |
| GPU 1 : ONNX + TensorRT FP32 | 1145,8 ± 2,6 [1142,9 ; 1149,3] | 0,0436 ± 0,0001 | 8,72 ± 0,02 [8,69 ; 8,74] | 0,92 | 0,060 |
| GPU 2 : torch.compile | 384,1 ± 3,2 [381,0 ; 389,1] | 0,1302 ± 0,0011 | 26,03 ± 0,21 [25,69 ; 26,24] | 0,92 | 0,179 |

Le Top-1 de 0,92 correspond à 46 bonnes prédictions sur 50 images, pour chacun des quatre pipelines.

## 4.2 Débit et latence

Le graphique de débit (barres d'erreur = écart-type) montre un écart majeur entre les deux plateformes. Les deux pipelines CPU plafonnent autour de 14 images/s, avec des barres d'erreur à peine visibles. Le pipeline GPU TensorRT FP32 atteint environ 1146 images/s, soit environ 80 fois le CPU. Le pipeline GPU torch.compile atteint 384 images/s, soit environ 27 fois le CPU et un tiers du débit de TensorRT.

Le graphique de latence par batch confirme ce résultat : une image de batch de 10 demande environ 712 ms sur CPU, contre 8,7 ms avec TensorRT et 26 ms avec torch.compile sur GPU.

Entre les deux pipelines CPU, la différence est négligeable : 14,06 contre 13,98 images/s, soit un écart de 0,6 %, inférieur à l'écart-type d'OpenVINO. Sur ce modèle et ce matériel, OpenVINO et Inductor produisent donc des performances équivalentes. OpenVINO est cependant plus variable (écart-type de 0,62 contre 0,23 pour torch.compile), alors que torch.compile a une latence plus régulière.

Sur GPU, TensorRT est nettement plus rapide que torch.compile. Cet écart s'explique par la nature des optimisations : TensorRT réalise une compilation statique du graphe avec un moteur figé pour une forme d'entrée donnée, ce qui supprime une grande partie de l'overhead Python et du lancement de kernels. Inductor génère des kernels Triton ou CUDA mais conserve une couche d'exécution PyTorch plus lourde.

## 4.3 Précision et accord des prédictions

Les quatre pipelines obtiennent le même Top-1 (0,92), avec une variance nulle entre les runs. La matrice d'accord des prédictions entre pipelines (notebook 05) affiche 1,00 pour toutes les paires : les pipelines produisent exactement les mêmes classes prédites sur les 50 images. Les optimisations (fusion d'opérateurs, changement de kernels, export ONNX) ne dégradent donc pas les prédictions dans ce cadre.

Cette conclusion doit être nuancée : 50 images ne permettent de distinguer que des écarts d'une image (2 points de pourcentage). Une comparaison plus fine nécessiterait de comparer les logits et non seulement les classes prédites.

## 4.4 Énergie

Sur GPU, la puissance moyenne mesurée est de 65,8 W pour TensorRT FP32 et de 67,6 W pour torch.compile. Malgré une puissance quasi identique, l'énergie par image diffère d'un facteur 3 : 0,060 J pour TensorRT contre 0,179 J pour torch.compile. Cela tient au débit : à puissance comparable, le pipeline le plus rapide consomme moins d'énergie pour traiter une même image.

Pour le CPU, l'énergie n'a pas pu être mesurée (compteur RAPL absent sur Colab). Les graphiques d'énergie ne comportent donc que deux barres.

## 4.5 Stabilité et reproductibilité

Tous les pipelines sont stables : l'écart-type relatif du débit reste inférieur à 0,5 % sur GPU (TensorRT : 2,6 sur 1146 ; torch.compile : 3,2 sur 384) et inférieur à 5 % sur CPU. Les mesures sont donc reproductibles au sein d'une même session.

Le pipeline CPU OpenVINO est le moins stable des pipelines CPU, avec une latence qui varie entre 682 et 755 ms selon les runs. Cette dispersion est cohérente avec un environnement partagé, où les cœurs CPU de la machine Colab sont sollicités par d'autres processus.

Les mesures n'ont pas été répétées sur plusieurs sessions Colab. La reproductibilité inter-sessions n'est donc pas établie.

# 5. Analyse et discussion

## 5.1 Compromis performance, précision et énergie

Les quatre pipelines donnent exactement les mêmes prédictions. Le choix se fait donc uniquement sur les performances et l'énergie. Pour une inférence en temps réel ou en grand volume sur GPU NVIDIA, TensorRT est le meilleur choix. Pour une inférence sur CPU, le gain de performance entre OpenVINO et torch.compile est nul sur ce modèle, et le choix peut reposer sur la facilité de déploiement ou sur la portabilité du format ONNX.

## 5.2 Coût de compilation

Les temps de compilation sont exclus des mesures, mais ils ne sont pas négligeables : environ 0,3 s pour OpenVINO, 33 s pour torch.compile sur CPU, 22 s pour torch.compile sur GPU, et 8 s pour le build TensorRT FP32. Ces coûts ne sont rentables que si le modèle sert pendant suffisamment longtemps pour amortir la compilation. Pour un déploiement où le modèle est rechargé souvent, ils doivent être pris en compte.

## 5.3 Limites et point ouvert : TensorRT FP16

Le pipeline TensorRT n'a été évalué qu'en FP32. La précision FP16, qui permettrait en principe d'accélérer l'inférence sur les GPU T4 grâce aux Tensor Cores, n'a pas pu être construite : l'API de la version de TensorRT installée sur Colab ne reconnaît plus le drapeau utilisé. Une correction consisterait à adapter la construction du réseau à la version installée (par exemple en exportant le modèle directement en FP16). Ce point reste à traiter pour compléter l'étude.

## 5.4 Autres limites

- Le modèle est un ResNet-18 avec une tête de classification réentraînée sur 2 époques seulement. Le Top-1 de 0,92 sur 50 images ne reflète pas la précision d'un modèle entraîné à fond.
- La mesure énergétique CPU est absente, et l'énergie GPU ne tient compte que de la puissance du GPU, pas de celle du processeur hôte.
- Les conditions matérielles de Colab (machines partagées, versions de bibliothèques) peuvent changer d'une session à l'autre.

# 6. Conclusion

Les quatre pipelines produisent des prédictions identiques, ce qui confirme que les optimisations n'ont pas dégradé la qualité sur cet échantillon. Le facteur déterminant est le matériel : le GPU est environ 27 à 80 fois plus rapide que le CPU selon le pipeline. Parmi les pipelines GPU, TensorRT en FP32 est 3 fois plus rapide que torch.compile et consomme 3 fois moins d'énergie par image. Sur CPU, OpenVINO et torch.compile sont équivalents en débit.

Ces conclusions sont établies sur un seul modèle, 50 images et une seule plateforme Colab. Elles donnent un ordre de grandeur fiable des écarts, mais une étude plus large devrait élargir l'échantillon, mesurer l'énergie CPU et compléter le pipeline TensorRT FP16.

# Références

- ONNX : https://onnx.ai
- OpenVINO Toolkit : https://docs.openvino.ai
- NVIDIA TensorRT : https://developer.nvidia.com/tensorrt
- PyTorch, torch.compile et TorchScript : https://pytorch.org/docs
- CIFAR-10 : Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition (ResNet).
