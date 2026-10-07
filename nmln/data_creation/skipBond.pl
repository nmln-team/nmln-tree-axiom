:- set_prolog_stack(global, limit(100000000000)).
:- set_prolog_stack(trail,  limit(20000000000)).
:- set_prolog_stack(local,  limit(2000000000)).

skipBond(X,Y) :- current_predicate(single/2), single(X,Z), single(Z,Y), X \= Y.

skipBond(X,Y) :- current_predicate(single/2), current_predicate(double/2), single(X,Z), double(Z,Y), X \= Y.
skipBond(X,Y) :- current_predicate(single/2), current_predicate(double/2), double(X,Z), single(Z,Y), X \= Y.
skipBond(X,Y) :- current_predicate(double/2), double(X,Z), double(Z,Y), X \= Y.

skipBond(X,Y) :- current_predicate(single/2), current_predicate(triple/2), single(X,Z), triple(Z,Y), X \= Y.
skipBond(X,Y) :- current_predicate(single/2), current_predicate(triple/2), triple(X,Z), single(Z,Y), X \= Y.
skipBond(X,Y) :- current_predicate(triple/2), triple(X,Z), triple(Z,Y), X \= Y.
skipBond(X,Y) :- current_predicate(double/2), current_predicate(triple/2), double(X,Z), triple(Z,Y), X \= Y.
skipBond(X,Y) :- current_predicate(triple/2), current_predicate(double/2), triple(X,Z), double(Z,Y), X \= Y.
