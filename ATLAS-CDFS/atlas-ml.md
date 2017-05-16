---
title: "Radio Galaxy Zoo: Crowdsourced labels for training machine learning methods for radio host cross-identification"
author: "M. J. Alger, J. K. Banfield, C. S. Ong, others"
geometry: margin=2cm
classoption: a4paper
latex-engine: xelatex
abstract: "We present a machine learning approach for radio host cross-identification, the task of determining the host galaxies of radio emissions detected in wide-area radio surveys. Training a machine learning algorithm requires a large amount [TODO: how large?] of labelled training data, which can be difficult or expensive to acquire. Radio Galaxy Zoo, a citizen science project on the Zooniverse platform, provides a large number of radio host cross-identifications which may be used as labels for training. However, these cross-identifications are assigned by non-experts, and may be incorrect. We find that while machine learning algorithms trained on expert labels outperform those trained on the crowdsourced Radio Galaxy Zoo labels, the accuracies are comparable and the crowdsourced labels are still useful for training. [TODO: add balanced accuracies]"
---

# Introduction

Next generation radio telescopes such as the Australian SKA Pathfinder (ASKAP; Johnston et al. 2007) and Apertif (Verheijen et al. 2008) will conduct increasingly wide, deep, and high-resolution surveys. These surveys will produce very large amounts of data. The Evolutionary Map of the Universe survey (EMU; Norris et al. 2011) using ASKAP and the WODAN survey (Röttgering et al. 2011) using Apertif are expected to detect over 100 million radio sources between them, compared to the 2.5 million radio sources already known (Banfield et al. 2015). An important part of processing this data is then cross-identifying observed radio emission regions with observations of the host galaxy that generated them in other wavelengths, usually infrared or optical [Julie?]. Radio host cross-identification is a difficult task, with radio-loud AGN often having complicated structures not clearly related to the host galaxy [TODO: examples? WAT?]. These AGN are expected to dominate approximately 30% of sources detected by EMU (Norris et al. 2011). Small surveys of a few thousand sources, such as the Australia Telescope Large Area Survey (ATLAS; Norris et al. 2006, Middelberg et al. 2008), can be cross-identified manually, but this is impractical for larger surveys. Automated algorithms exist (e.g. Fan et al. 2015) but these tend to be based on astrophysical models, and may fail for new classes of objects that could be found in surveys like EMU. Additionally, there will be so much data produced [TODO: how much?] by upcoming radio surveys that automated methods will be needed to choose what data to keep and what to discard. For these algorithms, robust feature selection is important and model-based approaches (e.g. Proctor 2006) will likely fail.

[TODO: literature review: paper Ivy sent, PINK, WTF, Fan, Proctor]

[TODO: Note that outlier detection can be done in lots of different ways including with this work and there are benefits in both. Also an example of where model-based approaches fall down. Elaborate on this?]

Radio Galaxy Zoo (RGZ; Banfield et al. 2015) is a citizen science project developed to help address the problem of radio host cross-identification for large radio surveys. It provides a large dataset of around 175&nbsp;000 [TODO(Julie): exact?] crowdsourced radio host cross-identifications for radio emissions detected in the Faint Images of the Radio Sky at Twenty-Centimeters survey (FIRST; White et al. 1997) and the Australia Telescope Large Area Survey (ATLAS; Franzen et al. 2015). Volunteers are presented with images from these surveys and a corresponding infrared image from the *Wide-Field Infrared Survey Explorer* (WISE; Wright et al. 2010) or the *Spitzer* Wide-Area Infrared Extragalactic Survey (SWIRE; Lonsdale et al. 2003), respectively. The volunteers are then asked to 1) identify which radio components belong to the same radio source, and 2) identify the associated infrared host galaxy for each source. [TODO(Julie): Provide numbers and explain why this is not efficient.]

In order to investigate methods for efficiently and effectively classifying radio galaxy hosts for upcoming large radio surveys, we have developed a machine learning approach for radio host cross-identification using standard machine learning techniques.

# Data

## Radio Galaxy Zoo
Radio Galaxy Zoo asks volunteers to cross-identify radio components in the ATLAS and FIRST surveys with their host galaxies in the SWIRE and WISE surveys. A more detailed description can be found in Banfield et al. (2015). In this paper we focus on the ATLAS and SWIRE surveys. There are two main reasons for this. The first is that ATLAS is the pilot study for EMU where automated methods like ours will be used. The second is that ATLAS is composed of two fields, so we can train methods on one field and test these methods on the other field to ensure that our methods apply in different areas of the sky. [TODO(Julie): Am I interpreting your comments on this correctly?]

[TODO: Define "component" and "source" and "primary component".]

Each primary component found in the ATLAS DR3 component catalogue appears in Radio Galaxy Zoo. Non-primary components may appear within the image of a primary component, but do not have their own entry in Radio Galaxy Zoo. We will henceforth only discuss the primary components. Each radio component is presented to volunteers as two overlayed images: one in radio, and one in infrared. The images are $2' \times 2'$ for ATLAS. Volunteers must first choose which radio components are associated with the same radio source. They are then able to choose the location the host galaxy in the infrared image. The coordinates of the pixel that they select as the host galaxy location are recorded and stored alongside the combination of radio components as a cross-identified radio source.

To reduce noise, multiple volunteers are presented with each radio object. Compact radio objects are shown to 5 volunteers, while extended radio objects are shown to 20 volunteers. These cross-identifications are combined to produce the final catalogue (see Wong et al., 2017). The host galaxy locations selected by volunteers who agree with the most commonly chosen combination of radio objects are combined by maximising over a kernel density estimate of the locations. These locations are then matched to the nearest SWIRE object within 5 arcseconds [TODO: check this].

## ATLAS

The Australia Telescope Large Area Survey (ATLAS; Franzen et al. 2013) is a wide-area radio survey of the *Chandra* Deep Field - South (CDFS) and the ESO Large Area ISO Survey - South 1 (ELAIS-S1) fields at 1.4 GHz. It is a pilot survey for the Evolutionary Map of the Universe (EMU; Norris et al. 2011) survey that will be conducted with ASKAP. EMU will cover the entire southern sky and is expected to detect approximately 70 million new radio sources. EMU will be conducted at the same depth and resolution as ATLAS, so methods developed for processing ATLAS data are expected to work for EMU. [TODO: How large are CDFS and ELAIS-S1? How many radio objects do they contain?] ATLAS has a sensitivity of 14 µJy on CDFS and 17 µJy on ELAIS-S1 (Franzen et al. 2013).

Norris et al. (2006) produced a catalogue of cross-identifications of 784 ATLAS radio objects with their infrared counterparts in the Spitzer Wide-area Infrared Extragalactic survey (SWIRE; Lonsdale et al. 2005?).

RGZ volunteers are asked to cross-identify objects in CDFS from ATLAS with their infrared counterparts in SWIRE, which has produced another catalogue of cross-identifications (Wong et al. 2017). As these cross-identifications have been based on non-expert classifications, this catalogue is expected to be lower-quality than an expert catalogue like that produced by Norris et al. (2006). [TODO: Discuss this paragraph with Julie]

## SWIRE

The Spitzer Wide-area Infrared Extragalactic survey is a wide-area infrared survey at the four IRAC wavelengths 3.6 µm, 4.5 µm, 5.8 µm, and 8.0 µm. It covers the eight SWIRE fields, particularly CDFS and ELAIS-S1, both of which were also covered by ATLAS. SWIRE is thus the source of infrared observations for cross-identification with ATLAS.

- noise levels

<!-- 
In this paper we focus on cross-identification of radio objects with their infrared counterparts in the Chandra Deep Field - South (CDFS) and ESO Large Area ISO Survey - South 1 (ELAIS-S1) fields. These fields have radio observations from the Australia Telescope Large Area Survey (ATLAS; Franzen et al. 2013) and infrared observations from both Spitzer (Lonsdale et al. 2005?) and WISE (????). -->

<!-- 
While radio objects may be manually cross-identified by expert astronomers, this is impractical with new, larger radio surveys that may detect millions of radio objects. Algorithms exist which automate this process using astrophysical models of how radio objects are expected to look (e.g. Proctor ????, Fan et al. 2015). However, with upcoming large radio surveys such as the Evolutionary Map of the Universe (EMU; Norris et al. 2011), these algorithms are expected to fail for 10% of new-found objects. -->

<!-- First, they must choose which radio components are part of the same radio source; second, they must identify this radio source with the infrared host galaxy. The hope is that this large database of cross-identifications can be used to train machine learning methods for cross-identifying objects in future surveys like EMU.
 -->

# Method

## Cross-identification as binary classification

We focus on the problem of cross-identification without reference to radio morphology. Given a radio component, we want to find its host galaxy as a citizen scientist would using Radio Galaxy Zoo. The input is thus a radio image and an infrared image, with radius $R$. We make the assumption that each radio image represents a single, complex extended source. This is not the case in general and a radio image may contain many different sources (e.g. RGZ uses cutouts where $R = 1'$, and these often contain multiple sources). We also note that this will bias our results against radio emission that extends beyond the $R$ cutout size. The radio cross-identification task then amounts to locating the host galaxy within the associated radio and infrared images. This is formalised as an object localisation problem: Given a radio image and an infrared image, locate the host galaxy.

[TODO: Rewrite to introduce the concept of a window. Start with the concept of rating each pixel, then use the window as a feature representation.] A common approach to such localisation problems is to slide a window across the image, centred on each pixel in turn. For each location the window visits, estimate the probability that the window contains the object we are trying to locate. The location with the highest probability is then assumed to be the location of the object. Applying this to radio cross-identification, we slide a 32 by 32 pixel window across a radio/infrared image and estimate the probability that the sliding window is centred on the host galaxy.

This task can be made greatly more efficient if we have a prior on the location of the object we are localising. For our prior, we assume that the host galaxy is always visible in the infrared and thus we only need consider windows centred on infrared sources. This assumption usually holds, except for a rare class of infrared-faint radio sources (Norris et al. 2006) [TODO: How many? I think it was 10]. This leads us to a binary classification task: Given an infrared source, compute the probability that it is a host galaxy. To find the host galaxy given a radio source, classify each galaxy within $R$ of the source and select the galaxy with the highest probability of being a host galaxy. This is a good formulation as binary classification is a very common problem in machine learning and there are many different methods readily available to solve it.

Solving the radio cross-identification task amounts to modelling a function $f$ from infrared sources $\mathcal{X}$ to binary $\mathcal{Y} = \{0, 1\}$:
$$
    f : \mathcal{X} \to \mathcal{Y}
$$
There are many options for modelling $f$. In this paper we apply three different models: logistic regression, random forests, and convolutional neural networks. As a linear method, logistic regression is the simplest classification approach we can take. Convolutional neural networks have recently shown strong results on image-based classification tasks. Random forests [something something].

The space of infrared sources $\mathcal{X}$ needs to be encoded as a vector for the models we will use. We describe this in [Section Representation].

The key problem with this approach is our assumption that the radio sky within radius $R$ contains only one, complete radio source. The problem is two-fold: This radius may contain multiple sources, or it may not contain the entirety of the source. If the radius contains multiple sources then there will also be multiple hosts in our input images (which breaks our assumption that there is only one); even a perfect classifier can only accurately cross-identify *one* host in an image with multiple. If the radius does not contain the whole source, then we are missing radio information useful for finding the host galaxy. This is a difficult problem in general (even for RGZ) as radio sources can be extremely wide. RGZ found a wide-angled tail radio source that spanned over three different images presented to volunteers and the full source was only cross-identified by the efforts of citizen scientists (Banfield et al. 2016). The problems are in opposition to each other: To reduce the number of sources in an input image, we can reduce $R$, but this increases the chance that we will miss relevant radio source information, and vice versa.

For our experiments, we take $R = 1'$, as in the images presented to RGZ volunteers. Our assumptions impose an upper bound on how well we can cross-identify radio sources, which can be estimated by considering how accurately a *perfect* binary classifier cross-identifies radio sources under our method. Using the Norris et al. (2006) labels (Section: Labels) to classify SWIRE objects to 100% accuracy on the binary classification task results in a cross-identification accuracy of 89% [method? where do I put this], which we take as an upper bound on our cross-identification accuracy.

It is unclear what size window to choose. If we choose a very large window, then we may get windows that contain multiple radio objects, or classifying may be computationally difficult. If we choose a very small window, then the window may not capture the whole radio source and we may not be able to accurately classify [TODO: see also ARG0003sky, the RGZ wide radio galaxy]

[TODO: Explicitly list assumptions and show examples of where they break. Will likely need to combine this with the above three paragraphs.]

[TODO: add a diagram of this pipeline since while it's a very common approach to object localisation it's not all that common in astrophysics]

## Vector representation of infrared sources

[TODO: Discuss comments on "vector" with Julie]

Most binary classification methods require that the inputs to be classified are real-valued vectors. We thus need to choose a vector representation of our candidate host galaxies, also known as the "features" of the galaxies.

[TODO: Add a plot of the SWIRE flux distributions along with the wedges]

Infrared observations of the CDFS field are taken from SWIRE. We use the CDFS Fall '05 SWIRE catalogue [cite] to generate candidate hosts to classify. Radio observations of the CDFS field are taken from ATLAS. As almost all infrared sources in SWIRE look essentially the same (and differences in structure are likely irrelevant to whether the galaxy is a host galaxy), the candidate host location encodes almost all information we could gain from the infrared image. We therefore only use the radio image for object localisation.

[TODO: Turn into bullet points.] We represent each candidate host as 1034 real-valued features. The first nine of these features are derived from the SWIRE catalogue: The logarithm of the ratio of fluxes in the four IRAC wavelengths, the stellarity index in both 3.6 µm and 4.5 µm, and the flux in 3.6 µm. The flux ratios are indicators of the star formation rate and amount of dust in the galaxy and might thus be predictors of whether the galaxy contains an AGN [Julie?] (*@fig:magdiff). the stellarity index represents how likely the object is to be a star rather than a galaxy, and the flux [why? Julie?]. The tenth feature is the distance across the sky between the candidate host and the nearest radio component in the ATLAS catalogue. The remaining 1024 features are the intensities of each pixel in a 32 $\times$ 32 pixel window centred on the candidate host. We chose to use a 32 $\times$ 32 pixel window as it seemed to give a good balance between performance and computational efficiency.

## Logistic regression

Logistic regression is the simplest classification model we can apply. It is linear in the feature space and outputs the probability that the input has a positive label. The model is

$$
    f(\vec x) = \sigma(\vec w \cdot \vec x + b)
$$
where $\vec w \in \mathbb{R}^D$ is a weights vector, $b \in \mathbb{R}$ is a bias term, $\vec x \in \mathbb{R}^D$ is the feature representation of a candidate host, and $\sigma$ is the logistic sigmoid function
$$
    \sigma(a) = (1 + \mathrm{exp}(-a))^{-1}.
$$

## Convolutional neural networks

Convolutional neural networks (CNNs) are a biologically-inspired prediction model for prediction with image inputs. A number of filters are convolved with the image to produce output images, and these outputs can then be convolved again with other filters on subsequent layers, producing a network of convolutions. This whole network is differentiable with respect to the values of the filters, and so the filters can be learned by gradient methods. The final layer of the network is logistic regression, with the convolved outputs as input features.

CNNs have recently produced good results on large image-based datasets, which is why we employ them in this paper. We employ only a simple model --- CNNs can be arbitrarily complex --- as this is a proof of concept.

[TODO: Add CNN diagram and example of a CNN reducing a radio image.]

## Random Forests

Random forests are an ensemble of decision trees. [todo]

## Labels

Converting the RGZ and Norris et al. (2006) cross-identification catalogues to binary labels for infrared objects is a non-trivial task. The most obvious problem is that there is no way to capture radio morphology information in binary classification. As a result, we ignore this problem for this paper. Another problem is that there is no way to indicate *which* radio object an infrared object is associated with, only that it is associated with *some* radio object. We make the naïve assumption that any given radio image contains only one host galaxy as the first method in solving this problem.

We then generate positive labels from a cross-identification catalogue. We decide that if an infrared object is listed in the catalogue, then it is assigned a positive label as a host galaxy. In principle we would then assign every other galaxy a negative label. This has some problems --- an example is that if the cross-identifier did not observe a radio object (e.g. it was below the signal-to-noise ratio) then the host galaxy of that radio object would receive a negative label. [TODO: Count how many times this happens in Norris/RGZ.] This occurs with Norris et al. (2006) cross-identifications, as these are associated with the first data release of ATLAS. The first data release went to a depth of [TODO: depth of DR1], compared to the depth of the third data release [TODO: depth of DR3] (Franzen et al. 2015) used by Radio Galaxy Zoo. Labels from Norris may therefore disagree with labels from Radio Galaxy Zoo even if they are both correct.

There are many potential host galaxies [TODO: how many galaxies in SWIRE?], so instead of using all galaxies in the CDFS field we only train and test our classifiers on infrared objects within a fixed radius $R$ of an ATLAS radio object. For this radius we choose $1'$, the same radius as the images shown to volunteers in RGZ. In general this will result in cases where the host galaxy is outside the radius (such as radio objects with wide-angled tails, e.g. Banfield et al. (2016)), but this is unavoidable. We may also choose a radius which is too *large*, worsening our assumption that there is only one host galaxy in this radius, but again, this is unavoidable.

[How badly does this assumption hurt us? How does a perfect binary classifier behave under these assumptions?]

(TODO: Describe how we assigned labels given that RGZ only labels the first component.)

## Experimental Setup

![CDFS field training and testing quadrants. The central dot is located at 52h48m00s -28°06m00s. There are similar numbers of radio sources in each quadrant.](quadrants.pdf){#fig:quadrants}

[TODO: Fix the axis labels.]

We trained cross-identifiers on radio objects from the ATLAS observations of the Chandra Deep Field - South (CDFS) using Radio Galaxy Zoo cross-identifications, and compared the trained cross-identifiers to those trained on a set of expert cross-identifications of the same field. We then applied the cross-identifiers trained on CDFS to the ESO Large Area ISO Survey - South 1 (ELAIS-S1) field [assuming we're still planning to do this].

We divided the CDFS field into four quadrants for training and testing. The quadrants were centred on 52h48m00s -28°06m00s (*@fig:quadrants). For each trial, one quadrant was used to draw test examples, and the other three quadrants were used for training examples.

We further divided the radio components into compact and resolved. Compact components are trivially cross-identified by fitting a Gaussian (as in Norris et al. 2006) and we would expect any machine learning approach for host cross-identification to attain high accuracy on this set. Whether a component was resolved was decided based on its flux; a radio component was considered resolved if
[TODO: Fix to follow Franzen; ensure pipeline is following Franzen too]
$$
    \frac{S_{\text{int}}}{S_{\text{peak}}} \leq 1
$$
and compact otherwise.

We considered only radio objects with a cross-identification in both the Norris et al. (2006) catalogue and the RGZ catalogue. Candidate hosts were then selected from the SWIRE catalogue. For a given subset of radio objects, all SWIRE objects within $R$ of all radio objects in the subset were added to the associated SWIRE subset.

Each classifier was trained on the training examples and used to predict labels for the test examples. The predicted labels were compared to the labels derived from the Norris et al. (2006) cross-identifications and the balanced accuracy was computed. We used balanced accuracy as our accuracy measure due to the highly imbalanced classes --- in our total set of SWIRE objects, only 4% have positive labels. The accuracies were then averaged across all four quadrants. Classification outputs are reported for the testing quadrant. We report the balanced accuracy on the classification task for logistic regression, convolutional neural networks, and random forests.

We then used the outputs of our classifiers to predict the host galaxy for each radio component cross-identified by both Norris et al. (2006) and Radio Galaxy Zoo. For each SWIRE object within 1 arcminute of the radio component, a probability of the object having a positive label was estimated using the trained binary classifiers. The SWIRE object with the highest probability was chosen as the host galaxy. The accuracy was then estimated by counting how many predicted host galaxies matched the Norris et al. (2006) cross-identifications.

The balanced accuracy provides a good proxy to the accuracy of the host cross-identification task. To check this, we selected random, small subsets of the SWIRE training objects, and for each random subset trained a logistic regression classifier. The balanced accuracy of this classifier was computed and compared with the accuracy of this classifier on the host cross-identification task. These accuracies are plotted against each other in Figure xxx. We found a positive correlation. Additionally, we compared the balanced accuracy of the classifier to the mean distance of the predicted host galaxy from the true host galaxy, also finding a positive correlation. [this is kind of a result, but it's used to describe why we chose the method we did to report --- should it go in the results and then discussion section, or here?]

# Results

[TODO: Include images of "good" and "bad" X-IDs/GCTs.]

![Balanced accuracies for each quadrant in the galaxy classification task.](atlas-ml-ba.pdf){#fig:ba}

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

![Classification balanced accuracy against accuracy on the cross-identification task. Cross-identification accuracy is computed from a binary comparison between the predicted host and the Norris et al. (2006) cross-identification; neither distance to the true host nor broken assumptions of one host per image are accommodated. [TODO: explain this caption better so that it is understandable to non-ML people]](gct-to-xid.pdf){#fig:gct-to-xid}

![Classification balanced accuracy against average radial distance[TODO(Julie): Is this correct?] between the predicted and the Norris et al. (2006) cross-identified host on the cross-identification task. ](gct-to-arcsec-error.pdf){#fig:gct-to-arcsec-error}

![Passive learning plot for the GCT. Trained and tested on RGZ. This is so that we had maximal training data — RGZ has many more objects than Norris.](passive.pdf){#fig:passive}

![Distribution of non-image features for different subsets.](distributions.pdf){#fig:distributions}

[TODO: Make the distribution plot actually exist]

![Colour-colour diagrams for each classifier.](colour_colour_predictions.pdf){#fig:colour-colour}

[TODO: Axis labels should say e.g. $\log_{10}(S_{5.8}/S_{3.6})$; include the SWIRE wedges from Lacy et al. 2004.]
<!-- 
![Density of colour-colour features for sampled SWIRE objects.](colour_colour_all.pdf)

![Density of colour-colour features for SWIRE objects with >95% probability of being a host galaxy according to logistic regression trained on RGZ.](colour_colour_95.pdf) -->