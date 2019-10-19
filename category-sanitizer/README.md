# Cleansing Wikipedia Categories using Centrality
## by Paolo Boldi and Corrado Monti

![Laboratory for Web Algorithmics](http://nexus.law.di.unimi.it/law-logo-transparent-small.png) 
---



We propose a novel general technique aimed at pruning and cleansing the Wikipedia category hierarchy, with a tunable level of aggregation. Our approach is endogenous, since it does not use any information coming from Wikipedia articles, but it is based solely on the user-generated (noisy) Wikipedia category folksonomy itself. 

For more information see [the paper, presented at WWW2016 (companion), Wiki Workshop 2016, at Montreal](http://dl.acm.org/ft_gateway.cfm?id=2891111&ftid=1707848).

# Provided dataset

We provide here a ready-to-use dataset, with a recategorization of the wikpedia pages to a set of 10 000 categories (the most important ones according to our approach). If you wish to use a different number of categories, please run the provided code.

_**To download the dataset go to [Releases](https://github.com/corradomonti/wikipedia-categories/releases/).**_ Here, you'll find:

* [`page2cat.tsv.gz`](https://github.com/corradomonti/wikipedia-categories/releases/download/enwiki-20160407/page2cat.tsv.gz) is a gzipped TSV file with the mapping from Wikipedia pages to cleansed categories, listed from the most important to the least important.
* [`ranked-categories.tsv.gz`](https://github.com/corradomonti/wikipedia-categories/releases/download/enwiki-20160407/ranked-categories.tsv.gz) is a gzipped TSV file with every Wikipedia category and our importance score.

We also provide the head of these files (`page2cat-HEAD.tsv` and `ranked-categories-HEAD.tsv`) to show how they look like after unzip.

If you wish to use the dataset or the code, please cite:
Paolo Boldi and Corrado Monti. "*Cleansing wikipedia categories using centrality.*" Proceedings of the 25th International Conference Companion on World Wide Web. International World Wide Web Conferences Steering Committee, 2016.

Bibtex:

    @inproceedings{boldi2016cleansing,
	    title={Cleansing wikipedia categories using centrality},
	    author={Boldi, Paolo and Monti, Corrado},
	    booktitle={Proceedings of the 25th International Conference Companion on World Wide Web},
	    pages={969--974},
	    year={2016},
	    organization={International World Wide Web Conferences Steering Committee}
    }


PLEASE NOTE: *Experiments described in the paper were run on a 2014 snapshot called
`enwiki-20140203-pages-articles.xml.bz2`, while – to provide an updated version –
this dataset refers to `enwiki-20160407-pages-articles.xml.bz2`.*

# How to run code

Set up the environment
----------------------

In order to compile the code, you'll need Java 8, Ant and Ivy. To install
them (e.g. inside a clean [Vagrant](http://vagrantup.com/) box with
`ubuntu/trusty64`), you should use these lines:

    sudo apt-get --yes update
    sudo apt-get install -y software-properties-common python-software-properties
    echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections
    sudo add-apt-repository ppa:webupd8team/java -y
    sudo apt-get update
    sudo apt-get --yes install oracle-java8-installer
    sudo apt-get --yes install oracle-java8-set-default
    sudo apt-get --yes install ant ivy
    sudo ln -s -T /usr/share/java/ivy.jar /usr/share/ant/lib/ivy.jar


Compile the code
----------------------

If the environment is set up properly, you should install git and download this repo with

	sudo apt-get install git
	git clone https://github.com/corradomonti/wikipedia-categories.git

and then go to the directory `java`. There, run:

* `ant ivy-setupjars` to download dependencies
* `ant` to compile
* `. setcp.sh` to include the produced jar inside the Java classpath.

Now you are ready to run `run.sh`, which will assume to have the file `WIKIDUMP_XML` as `enwiki-20160407-pages-articles.xml.bz2`.
