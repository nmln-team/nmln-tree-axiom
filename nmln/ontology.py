import numpy as np
import typing as t
import networkx as nx

from collections import OrderedDict
from itertools import product, combinations, permutations

from nmln import utils, parser


class Domain:
    def __init__(
        self,
        name: str,
        constants: t.List[str],
        features: t.Optional[np.ndarray] = None,
    ):
        """
        Class of objects representing a Domain of domains in a FOL theory

        Args:
            name: name of the domain
            constants: list of domains identifiers (strings or integers)
            features: np.array, a matrix [num_constants, feature_size], where each row i represents the features of constant i
        """
        self.name = name
        self.constants = constants

        """Map domains names to id (row index)"""
        self.constant_name_to_id = {c: i for i, c in enumerate(self.constants)}

        if features is not None:
            assert features.shape[0] == len(constants)
            self.features = features

        else:
            self.features = np.expand_dims(
                np.eye(self.num_constants, dtype=np.float32), 0
            )

    @property
    def num_constants(self) -> int:
        return len(self.constants)

    def __hash__(self) -> int:
        return hash(self.name)


class Predicate:
    def __init__(
        self,
        name: str,
        domains: t.List[Domain],
    ):
        """
        Class of objects representing a relations in a FOL theory.

        Args:
            name: the unique name of the predicate
            domains: A positional list of domains
        """
        self.name = name

        self.domains = domains
        self.groundings_number = self.calc_groundings_number(self.domains)

    @staticmethod
    def calc_groundings_number(domains: t.List[Domain]) -> int:
        groundings_number = 1

        for domain in domains:
            groundings_number *= domain.num_constants

        return groundings_number

    @property
    def arity(self) -> int:
        return len(self.domains)

    def __lt__(self, other: "Predicate") -> bool:
        return self.name < other.name


class Ontology:
    def __init__(
        self,
        domains: t.List[Domain],
        predicates: t.List[Predicate],
    ):
        """
        The central object. It represents a multi-sorted FOL language.

        Args:
            domains: the domains of the language
            predicates: the predicates of the language
        """
        self.domains = {}
        self._domain_list = []
        self.predicates = OrderedDict()
        self.herbrand_base_size = 0
        self._predicate_range = OrderedDict()
        self._range_to_predicate = utils.RangeBisection()
        self.finalized = False
        self.constraints = []
        # Overall, number of elements in the assignment vector.
        self.linear_size = 0

        for d in domains:
            self.__add_domain(d)

        if len(domains) == 1:
            self.num_constants = domains[0].num_constants

        self.tuple_indices = {}
        for p in predicates:
            self.__add_predicate(p)

        self.__create_indexing_scheme()

        """ For some datasets, computing the indices of fragments is heavy. We store them."""
        self.all_fragments_cache = {}

    def __str__(self):
        s = ""
        s += (
            "Domains (%d): " % len(self.domains)
            + ", ".join(
                [
                    "%s (%d)" % (name, domain.num_constants)
                    for name, domain in self.domains.items()
                ]
            )
            + "\n"
        )
        s += (
            "Predicates (%d):" % len(self.predicates)
            + ", ".join(self.predicates.keys())
            + "\n"
        )
        return s

    def __add_domain(self, domain: Domain):
        if domain.name in self.domains:
            raise Exception("Domain %s already exists" % d.name)

        self.domains[domain.name] = domain
        self._domain_list.append(domain)

    def __add_predicate(self, predicate: Predicate):
        if predicate.name in self.predicates:
            raise Exception("Predicate %s already exists" % predicate.name)

        self.predicates[predicate.name] = predicate
        self._predicate_range[predicate.name] = (
            self.herbrand_base_size,
            self.herbrand_base_size + predicate.groundings_number,
        )
        self._range_to_predicate[
            (
                self.herbrand_base_size,
                self.herbrand_base_size + predicate.groundings_number - 1,
            )
        ] = predicate.name
        self.herbrand_base_size += predicate.groundings_number
        k = tuple([d.name for d in predicate.domains])
        if k not in self.tuple_indices:
            # Cartesian product of the domains.
            ids = np.array(
                [i for i in product(*[range(self.domains[d].num_constants) for d in k])]
            )
            self.tuple_indices[k] = ids

    def __create_indexing_scheme(self):
        """
        Creates the indexing scheme used by the Ontology object for all the logic to tensor operations.
        """
        # Managing a linearized version of this logic
        self._up_to_idx = 0  # linear max indices
        self._dict_indices = (
            {}
        )  # mapping potentials id to correspondent multidimensional indices tensor

        self.finalized = False
        self._linear = None
        self._linear_evidence = None

        # Overall, number of elements in the assignment vector.
        self.linear_size = 0
        for p in self.predicates.values():
            # For unary predicates, this is just the domain size as [size]
            # For n-ary predicates, this is just the tensor of domain sizes [d1_size, d2_size, ...]
            shape = [d.num_constants for d in p.domains]
            # Overall domain size.
            predicate_domain_size = np.prod(shape)
            start_idx = self._up_to_idx
            end_idx = start_idx + predicate_domain_size
            self._up_to_idx = end_idx
            # print('Dict Indices', start_idx, end_idx)
            self._dict_indices[p.name] = np.reshape(
                np.arange(start_idx, end_idx), shape
            )
            self.linear_size += predicate_domain_size
        self.linear_size = int(self.linear_size)
        self.finalized = True

    def file_content_to_linearState(self, content: t.List[str]) -> np.ndarray:
        state = np.zeros(self.linear_size)
        ids = [self.atom_string_to_id(line) for line in content]

        state[ids] = 1

        return state

    def file_to_linearState(self, file: str) -> np.ndarray:
        state = np.zeros(self.linear_size)

        with open(file) as f:
            ids = [self.atom_string_to_id(line) for line in f]

        state[ids] = 1

        return state

    def linear_to_fol_dictionary(self, linear_state):
        """
            Create a dictionary mapping predicate names to np.array. For each key-value pair, the "value" of the
            dictionary array is the adiacency matrix of the predicate with name "key".

        Args:
            linear_state: a np.array with shape [self.linear_size]

        Returns:
            a dictionary mapping predicate names to np.array

        """
        d = OrderedDict()

        for p in self.predicates.values():
            d[p.name] = np.take(linear_state, self._dict_indices[p.name])

        return d

    def linear_to_networkx_graph(self, linear_state):
        fol_dict = self.linear_to_fol_dictionary(linear_state)
        assert (
            "c1" in fol_dict and "single1" in fol_dict
        ), "TODO: Complete this function."
        try:
            graph = nx.from_numpy_array(fol_dict["single1"].numpy())
        except AttributeError:
            graph = nx.from_numpy_array(fol_dict["single1"])
        return graph

    def atom_string_to_id(self, atom):
        predicate, constants = parser.atom_parser(atom)
        p = self.predicates[predicate]
        constants_ids = []
        for i, c in enumerate(constants):
            try:
                constants_ids.append(p.domains[i].constant_name_to_id[c])
            except Exception as e:
                raise Exception(
                    f"Predicate {predicate} does not have a constant {c} in domains {p.domains}." 
                ) from e
        return self.atom_to_id(predicate, tuple(constants_ids))

    def atom_to_id(self, predicate_name, constant_ids):
        return self._dict_indices[predicate_name][tuple(constant_ids)]

    def nested(
        self, array: list, k: int, max_k: int, num_constants: int, idx: list = []
    ) -> list:
        for i in range(num_constants):
            array.append([])
            current_idx = idx + [i]

            if k < max_k:
                self.nested(
                    array[-1],
                    k=k + 1,
                    max_k=max_k,
                    num_constants=num_constants,
                    idx=current_idx,
                )

            else:
                for p in self.predicates.values():
                    f_idx = self._dict_indices[p.name]
                    for j in range(p.arity):
                        f_idx = np.take(f_idx, current_idx, axis=j)
                    f_idx = np.reshape(f_idx, [-1])

                    array[-1].extend(f_idx)

        return array

    def all_fragments_idx_quantifiers(self, k: int):
        num_constants = list(self.domains.values())[0].num_constants
        array = np.array(self.nested([], 1, max_k=k, num_constants=num_constants))
        return array

    def all_fragments_idx(self, k: int):
        ii = []

        num_constants = list(self.domains.values())[0].num_constants

        for fragments in combinations(range(num_constants), k):
            ii.append([])

            for idx in permutations(fragments, len(fragments)):
                i = []

                for p in self.predicates.values():
                    f_idx = self._dict_indices[p.name]
                    for j in range(p.arity):
                        f_idx = np.take(f_idx, idx, axis=j)
                    f_idx = np.reshape(f_idx, [-1])
                    i.extend(f_idx)

                ii[-1].append(i)

        return np.array(ii)


def load_predicates(ontology_path: str) -> dict:
    predicates = {}

    with open(ontology_path) as f:
        for line in f:
            if not line.strip():
                continue
            k, v = line.split(":")
            predicates[k] = int(v)

    return predicates
