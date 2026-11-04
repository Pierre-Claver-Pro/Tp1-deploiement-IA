# TP Déploiement de modèles d'IA : comparaison de pipelines d'inférence

Comparaison de quatre chaînes de déploiement d'un même modèle PyTorch (ResNet-18, 10 classes, CIFAR-10) sur CPU et GPU, selon les couches : haut niveau, format du modèle, middle level, bas niveau, matériel.

## Pipelines

| # | Plateforme | Chaîne | Notebook |
|---|---|---|---|
| 1 | CPU | PyTorch -> ONNX -> OpenVINO Runtime -> CPU | `notebooks/01_cpu_openvino.ipynb` |
| 2 | CPU | PyTorch -> torch.compile (Inductor) -> CPU (variante TorchScript incluse) | `notebooks/02_cpu_torchcompile.ipynb` |
| 3 | GPU | PyTorch -> ONNX -> TensorRT (FP32 et FP16) -> CUDA/cuDNN -> GPU | `notebooks/03_gpu_tensorrt.ipynb` |
| 4 | GPU | PyTorch -> torch.compile (Inductor) -> CUDA -> GPU | `notebooks/04_gpu_torchcompile.ipynb` |

## Protocole expérimental

- **Données :** 50 images tirées du jeu de test CIFAR-10 (graine 42), identiques pour les quatre pipelines.
- **Prétraitement :** redimensionnement 32x32 vers 224x224, normalisation ImageNet (moyenne `[0.485, 0.456, 0.406]`, écart-type `[0.229, 0.224, 0.225]`).
- **Modèle :** ResNet-18 initialisé sur ImageNet, tête remplacée par une couche à 10 sorties, puis fine-tuné 2 époques sur 10 000 images de CIFAR-10 (`notebooks/00_preparation.ipynb`).
- **Batch :** 10 images, identique pour tous les pipelines.
- **Répétitions :** 3 passes de chauffe, puis 5 runs indépendants de 5 passes chacun. Moyenne, écart-type, minimum et maximum sont calculés pour le débit, le temps total, la latence par batch et le Top-1.
- **Énergie :** NVML sur GPU (puissance moyenne sur 10 s, multipliée par la durée). RAPL sur CPU lorsque disponible.

## Organisation du dépôt

```
.
├── README.md
├── notebooks/        # 00 préparation, 01 à 04 pipelines, 05 comparaison
├── src/common.py     # chargement des données, modèle, benchmark, mesure d'énergie
├── results/          # JSON des mesures, une entrée par pipeline
├── figures/          # graphiques exportés depuis 05_comparaison
└── rapport/          # rapport technique (Markdown source et PDF)
```

## Exécution

1. Créer le dossier `MyDrive/tp_deploy` sur Google Drive et y copier `src/common.py`.
2. Exécuter `00_preparation.ipynb` une fois, en runtime GPU. Il crée `resnet18_cifar10.pt` et `inputs.pt` dans le dossier Drive.
3. Exécuter `01` et `02` en runtime CPU (Colab, "Aucun accélérateur").
4. Exécuter `03` et `04` en runtime GPU (T4).
5. Exécuter `05_comparaison.ipynb` pour générer les tableaux et les graphiques. Les JSON sont lus depuis `tp_deploy/results/`.
6. Copier les JSON dans `results/` et les figures dans `figures/` du dépôt.

## Résultats principaux

| Pipeline | Débit (img/s) | Latence batch (ms) | Top-1 | Énergie (J/img) |
|---|---|---|---|---|
| CPU OpenVINO | 14,06 ± 0,62 | 712 ± 32 | 0,92 | non mesurée |
| CPU torch.compile | 13,98 ± 0,23 | 716 ± 12 | 0,92 | non mesurée |
| GPU TensorRT FP32 | 1145,8 ± 2,6 | 8,72 ± 0,02 | 0,92 | 0,060 |
| GPU torch.compile | 384,1 ± 3,2 | 26,0 ± 0,2 | 0,92 | 0,179 |

Détails et analyse dans `rapport/rapport.md`.

## Conclusions

- Les quatre pipelines donnent exactement les mêmes prédictions (accord de 1,00 entre pipelines) : les optimisations ne dégradent pas la qualité sur cet échantillon.
- Le GPU est 27 à 80 fois plus rapide que le CPU selon le pipeline.
- Sur GPU, TensorRT FP32 est environ 3 fois plus rapide que torch.compile et consomme 3 fois moins d'énergie par image.
- Sur CPU, OpenVINO et torch.compile ont un débit équivalent.

## Limites

- Mesures effectuées sur Google Colab (machines partagées, variabilité matérielle possible d'une session à l'autre).
- Échantillon de 50 images : le Top-1 a une précision limitée (une image vaut 2 points).
- Fine-tuning rapide de la tête du modèle : le Top-1 ne reflète pas celui d'un modèle entraîné complètement.

## Auteur

Pierre-Claver Olivier KOFFI
