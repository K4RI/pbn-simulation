FunctionHood: Compute direct neighbours of any monotone Boolean Function
=========================================================

This project aims to, given a Boolean function, compute its direct neighbours in the Partially Ordered Set of monotone Boolean functions, without the need to generate the whole set. The set-representation is used to specify monotone Boolean functions.


How to use it?
--------------

To compile it you will need is java6 JDK and [maven](http://maven.apache.org/).

* grab the source from github
* compile and package it: `mvn package`
* you can use the jar in the "target" subdirectory: `cd target/`

You can either integrate the .jar file in your tool, or launch the GUI to compute immediate neighbour functions, as follows:

    java -jar FunctionHood.jar

Licence
-------

This code is available under GPL-3.0.


Cite
----

This work is available on arXiv: [https://arxiv.org/abs/1901.07623](https://arxiv.org/abs/1901.07623).


Authors
-------

José E. R. Cury [http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4787745U6](http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4787745U6)

Pedro T. Monteiro [http://pedromonteiro.org/](http://pedromonteiro.org/)

Claudine Chaouiya [https://claudine-chaouiya.pedaweb.univ-amu.fr/](https://claudine-chaouiya.pedaweb.univ-amu.fr/)
