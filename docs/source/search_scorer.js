const Scorer = {
  // API documentation is most useful when an exact component/type search lands
  // on that object's own entry before pages which merely mention it.
  objNameMatch: 50,
  objPartialMatch: 20,

  // Preserve Sphinx's normal ordering between documented-object priorities.
  objPrio: {
    0: 15,
    1: 5,
    2: -5,
  },
  objPrioDefault: 0,

  // Keep the standard weights for ordinary page-title and text matches.
  title: 15,
  partialTitle: 7,
  term: 5,
  partialTerm: 2,
};
