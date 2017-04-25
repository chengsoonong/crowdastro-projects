---
title: "Radio Galaxy Zoo: Crowdsourced labels for training machine learning methods for radio cross-identification"
author: "M. J. Alger, J. K. Banfield, C. S. Ong"
geometry: margin=3cm
latex-engine: xelatex
---
# Radio cross-identification

Radio cross-identification is the task of associating objects detected in wide-area radio surveys with the corresponding object in other wavelengths. In this paper we focus on cross-identification of radio objects with their infrared counterparts in the Chandra Deep Field - South (CDFS) and ESO Large Area ISO Survey - South 1 (ELAIS-S1) fields. These fields have radio observations from the Australia Telescope Large Area Survey (ATLAS; Franzen et al. 2013) and infrared observations from both Spitzer (Lonsdale et al. 2005?) and WISE (????).

While radio objects may be manually cross-identified by expert astronomers, this is impractical with new, larger radio surveys that may detect millions of radio objects. Algorithms exist which automate this process using astrophysical models of how radio objects are expected to look (e.g. Proctor ????, Fan et al. 2015). However, with upcoming large radio surveys such as the Evolutionary Map of the Universe (EMU; Norris et al. 2011), these algorithms are expected to fail for 10% of new-found objects.

Radio Galaxy Zoo (Banfield et al. 2015) was developed to help address this problem by generating a set of cross-identifications for 175000 radio objects in ATLAS and Faint Images of the Radio Sky at Twenty-Centimeters (FIRST; ????). Volunteers are presented with an image from ATLAS or FIRST and the corresponding infrared image from Spitzer or WISE, respectively, and are asked to perform two tasks. First, they must choose which radio components are part of the same radio source; second, they must identify this radio source with the infrared host galaxy. The hope is that this large database of cross-identifications can be used to train machine learning methods for cross-identifying objects in future surveys like EMU.

To reduce noise, multiple volunteers are presented with each radio object; compact radio objects are shown to 5 volunteers, and extended radio objects are shown to 20 volunteers. These cross-identifications are combined to produce the final catalogue. The host galaxy locations selected by volunteers who agree with the plurality combination of radio objects are combined by maximising over the kernel density estimate of the locations. These locations are then matched to the nearest SWIRE object within 5 arcseconds (check this).

# ATLAS

The Australia Telescope Large Area Survey (ATLAS; Franzen et al. 2013) is a wide-area radio survey of CDFS and ELAIS-S1 at 1.4 GHz. It is a pilot survey for the Evolutionary Map of the Universe (EMU; Norris et al. 2011) survey that will be conducted with the Australian SKA Pathfinder (ASKAP) telescope. EMU will cover the entire southern sky and is expected to detect around 70 million new radio sources. EMU will be conducted at the same depth and resolution as ATLAS, so methods developed for processing ATLAS data are expected to work for EMU. ATLAS has a sensitivity of 14 µJy on CDFS and 17 µJy on ELAIS-S1.

Norris et al. (2006) produced a catalogue of cross-identifications of 784 ATLAS radio objects with their infrared counterparts in the Spitzer Wide-area Infrared Extragalactic survey (SWIRE; Lonsdale et al. 2005?).

RGZ volunteers are asked to cross-identify objects in CDFS from ATLAS with their infrared counterparts in SWIRE, which has produced another catalogue of cross-identifications (Wong et al. 2017). As these cross-identifications have been based on non-expert classifications, this catalogue is expected to be lower-quality than an expert catalogue like that produced by Norris et al. (2006).

# SWIRE

The Spitzer Wide-area Infrared Extragalactic survey is a wide-area infrared survey at 3.6 µm, 4.5 µm, 5.8 µm, and 8.0 µm. It covers the eight SWIRE fields, particularly CDFS and ELAIS-S1, both of which were also covered by ATLAS. SWIRE is thus the source of infrared observations for cross-identification with ATLAS.

- noise levels

# Cross-identification as binary classification
We focus on the problem of cross-identification without reference to radio morphology. Given a radio image from RGZ/ATLAS, we assume that the image represents a single, complex extended source. The radio cross-identification task then amounts to locating the host galaxy within the associated radio and infrared images, just as a RGZ volunteer would do. This is formalised as an object localisation problem: Given a radio image, locate the host galaxy.

A common approach to such localisation problems is to slide a window across each pixel in the image. For each location the window visits, estimate the probability that the window contains the object we are trying to locate. The location with the highest probability is then assumed to be the location of the object. Applying this to radio cross-identification, we slide a 32 by 32 pixel window across an image from RGZ/ATLAS and estimate the probability that the sliding window is centred on the host galaxy.

This task can be made greatly more efficient if we have a prior on the location of the object we are localising; for this prior, we assume that the host galaxy is always visible in the infrared and thus we only need consider windows centred on infrared sources. This assumption usually holds, except for a rare class of infrared-faint radio sources (Norris et al. 2006). This leads us to a binary classification task: Given an infrared source, compute the probability that it is a host galaxy. This is a good formulation as binary classification is a very common problem in machine learning and there are many different methods readily available to solve it.

Infrared observations of the CDFS field come from SWIRE. We use the CDFS SWIRE catalogue to generate candidate hosts to classify.

Solving the radio cross-identification task amounts to modelling a function from infrared sources to binary:
$$
    y : \mathcal{IR} \to \mathbb{Z}_2
$$

There are many options for modelling $y$. In this paper we apply two different models: logistic regression and convolutional neural networks. As a linear method, logistic regression is the simplest classification approach we can take. Convolutional neural networks have recently shown strong results on image-based classification tasks.

## Vector representation of infrared sources

Most binary classification methods require that the inputs to be classified are real-valued vectors. We thus need to choose a vector representation of our candidate host galaxies, also known as the "features" of the galaxies.

![Magnitude differences may be predictors for whether a galaxy is a host galaxy. Reproduced from Banfield et al. (2015).](magdiff.pdf)

We represent each candidate host as 1029 real-valued features. The first four of these features are taken from the SWIRE catalogue: The difference between the 3.6 µm and 4.5 µm magnitudes, the difference between the 4.5 µm and 5.8 µm magnitudes, and the stellarity index in both 3.6 µm and 4.5 µm. The magnitude differences are indicators of the star formation rate and amount of dust in the galaxy and might thus be predictors of whether the galaxy contains an AGN, and the stellarity index represents how likely the object is to be a star rather than a galaxy. The fifth feature is the distance across the sky between the candidate host and the nearest radio component in the ATLAS catalogue. The remaining 1024 features are the intensities of each pixel in a 32 x 32 pixel window centred on the candidate host.

## Logistic regression

Logistic regression is the simplest classification model we can apply. It is linear in the feature space and outputs the probability that the input has a positive label. The model is

$$
    y(\vec x) = \sigma(\vec w \cdot \vec x + b)
$$
where $\vec w \in \mathbb{R}^D$ is a weights vector, $b \in \mathbb{R}$ is a bias term, $\vec x \in \mathbb{R}^D$ is the feature representation of a candidate host, and $\sigma$ is the logistic sigmoid function
$$
    \sigma(a) = (1 + \mathrm{exp}(-a))^{-1}.
$$

## Convolutional neural networks

Convolutional neural networks (CNNs) are a biologically-inspired prediction model for prediction with image inputs. A number of filters are convolved with the image to produce output images, and these outputs can then be convolved again with other filters on subsequent layers, producing a network of convolutions. This whole network is differentiable with respect to the values of the filters, and so the filters can be learned by gradient methods. The final layer of the network is logistic regression, with the convolved outputs as input features.

CNNs have recently produced good results on large image-based datasets, which is why we employ them in this paper. We employ only a simple model --- CNNs can be arbitrarily complex --- as this is a proof of concept.

## Labels

Converting the RGZ and Norris et al. (2006) cross-identification catalogues to binary labels for infrared objects is a non-trivial task. The most obvious problem is that there is no way to capture radio morphology information in binary classification; we ignore this problem for this paper. Another problem is that there is no way to indicate *which* radio object an infrared object is associated with, only that it is associated with *some* radio object. We make the (incorrect) assumption that any given RGZ/ATLAS image contains only one host galaxy, and defer solving this problem.

Generating positive labels from a cross-identification catalogue is simple: If an infrared object is mentioned in the catalogue, then it is a host galaxy, and is assigned a positive label. In principle we would then assign every other galaxy a negative label. This has some problems --- an example is that if the cross-identifier did not observe a radio object (perhaps it was too faint) then the host galaxy of that radio object would receive a negative label. This indeed happens with Norris cross-identifications, where not all objects in the third data release of ATLAS (this is the data release associated with RGZ) were observed, and hence labels from Norris may disagree with labels from RGZ even if they are both correct.

There are a lot of galaxies (citation needed), so instead of using all galaxies in the CDFS field we only train and test our classifiers on infrared objects within a fixed radius of an ATLAS radio object. For this radius we choose 1 arcminute, the same radius as the images shown to volunteers in RGZ. In general this will result in cases where the host galaxy is outside the radius (such as radio objects with wide-angled tails, e.g. Banfield et al. (2016)), but this is unavoidable. We may also choose a radius which is too *large*, worsening our assumption that there is only one host galaxy in this radius, but again, this is unavoidable.

# Method

We divided the CDFS field into four quadrants for training and testing. The quadrants were centred on 52.8h -28.1°. For each trial, one quadrant was used to draw test examples, and the other three quadrants were used for training examples.

We considered only radio objects with a cross-identification in both the Norris et al. (2006) catalogue and the RGZ catalogue. We further divided this subset into compact objects and resolved objects. Candidate hosts were then selected from the SWIRE catalogue; for a given subset of radio objects, all SWIRE objects within 1 arcminute of all radio objects in the subset were added to the associated SWIRE subset.

Each classifier was trained on the training examples and used to predict labels for the test examples. The predicted labels were compared to the labels derived from the Norris et al. (2006) cross-identifications and the balanced accuracy was computed. The accuracies were then averaged across all four quadrants. Classification outputs are reported for the testing quadrant. In addition to logistic regression and convolutional neural networks, we report the balanced accuracy for random forest classifiers on the classification task, as these are a common classifier in astrophysics.

# Results

![Balanced accuracies for each quadrant in the galaxy classification task.](atlas-ml-ba.pdf)

| Classifier | Training labels | Compact or resolved | Balanced accuracy (%) |
|------------|-----------------|---------------------|-----------------------|
| LR         | Norris          | Compact             | 94.7 $\pm$ 1.4        |
| LR         | RGZ             | Compact             | 95.1 $\pm$ 0.7        |
| LR         | Norris          | Resolved            | 89.9 $\pm$ 1.7        |
| LR         | RGZ             | Resolved            | 87.8 $\pm$ 4.5        |
| LR         | Norris          | Both                | 94.7 $\pm$ 1.4        |
| LR         | RGZ             | Both                | 91.9 $\pm$ 1.1        |

Table: Balanced accuracies for the galaxy classification task. Uncertainties represent standard deviation over the quadrants.

| swire | lr(rgz) | lr(norris) |
|-------|---------|------------|
| SWIRE3_J032559.15-284724.2 | 0.0341784863105 | 0.015480243078  |
| SWIRE3_J032559.91-284728.9 | 0.278541709304  | 0.020090942017  |
| SWIRE3_J032600.02-284736.9 | 0.245365593177  | 0.014413236572  |
| SWIRE3_J032600.13-284637.5 | 0.0813282413296 | 0.0208829692218 |
| SWIRE3_J032600.13-284715.7 | 0.387394784166  | 0.0343210418749 |
| SWIRE3_J032600.98-284705.4 | 0.145593835335  | 0.0658117444017 |
| SWIRE3_J032601.03-284711.6 | 0.677611173993  | 0.131620806718  |
| SWIRE3_J032601.75-284614.5 | 0.134551589362  | 0.0131522724495 |
| SWIRE3_J032602.08-284713.1 | 0.741262952211  | 0.565229482364  |

Table: Predicted probabilities for each SWIRE object. Predictors are logistic regression trained on RGZ labels and logistic regression trained on Norris labels. SWIRE objects that do not appear in the table have no prediction; we assume these are 0. Full table electronic.

| zooniverse_id | ra | dec | lr(rgz)_swire | lr(norris)_swire | rgz_swire | rgz_consensus_radio_level | rgz_consensus_ir_level |
|-|--|---|--------|--------|------|----------|---------|
| ARG0003rb2 | 51.511734 | -28.785575 | SWIRE3_J032602.36-284711.5 | SWIRE3_J032602.36-284711.5 | -99 | 0.4 | 0.333333333333 |
| ARG0003rfr | 51.564555 | -28.774847 | SWIRE3_J032615.41-284630.7 | SWIRE3_J032615.41-284630.7 | SWIRE3_J032616.14-284552.9,SWIRE3_J032615.41-284630.7 | 0.3125 | 1.0,1.0 |
| ARG0003r8s | 51.564799 | -28.099955 | SWIRE3_J032615.52-280559.8 | SWIRE3_J032615.52-280559.8 | SWIRE3_J032616.71-280538.6,SWIRE3_J032617.94-280648.2,SWIRE3_J032615.52-280559.8 | 0.484848484848 | 0.75,0.555555555556,0.8125 |
| ARG0003r2j | 51.572279 | -28.119491 | SWIRE3_J032615.86-280628.8 | SWIRE3_J032617.02-280638.9 | SWIRE3_J032617.89-280707.2 | 0.421052631579 | 1.0 |
| ARG0003raz | 51.604711 | -28.152731 | SWIRE3_J032625.19-280910.1 | SWIRE3_J032625.19-280910.1 | SWIRE3_J032624.80-280915.9 | 0.3 | 0.333333333333 |
| ARG0003ro4 | 51.621251 | -28.113924 | SWIRE3_J032629.13-280650.7 | SWIRE3_J032629.13-280650.7 | SWIRE3_J032629.13-280650.7,SWIRE3_J032626.74-280636.7 | 0.357142857143 | 0.8,1.0 |
| ARG0003r8e | 51.623385 | -28.681315 | SWIRE3_J032629.54-284051.9 | SWIRE3_J032629.54-284051.9 | SWIRE3_J032629.54-284055.8 | 0.3 | 1.0 |
| ARG0003r3w | 51.624653 | -28.798195 | SWIRE3_J032629.81-284754.4 | SWIRE3_J032629.81-284754.4 | SWIRE3_J032629.81-284754.4 | 1.0 | 0.666666666667 |
| ARG0003r55 | 51.62777 | -28.615917 | SWIRE3_J032630.64-283658.0 | SWIRE3_J032630.64-283658.0 | SWIRE3_J032630.64-283658.0,SWIRE3_J032628.56-283744.8 | 0.354838709677 | 1.0,0.727272727273 |
| ARG0003rj2 | 51.644117 | -28.339678 | SWIRE3_J032634.58-282022.8 | SWIRE3_J032634.58-282022.8 | SWIRE3_J032630.21-282025.5,SWIRE3_J032634.58-282022.8,SWIRE3_J032631.96-281941.0 | 0.59375 | 0.684210526316,0.947368421053,0.473684210526 |

Table: Predicted SWIRE hosts for ATLAS radio objects. Note the assumption that there is only one host galaxy per Zooniverse ID. Full table electronic.

![Classification balanced accuracy against accuracy on the cross-identification task. Cross-identification accuracy is computed from a binary comparison between the predicted host and the Norris et al. (2006) cross-identification; neither distance to the true host nor broken assumptions of one host per image are accommodated.](gct-to-xid.pdf)