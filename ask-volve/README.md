# Ask Volve
Build a conversational assistant that makes the Volve field dataset easier to explore.

## Problem

Equinor has openly released the complete dataset from the Volve field: roughly 40,000 files from a
real North Sea oil field covering 2008 to 2016. The dataset is comprehensive, but common questions
are difficult to answer because information is spread across spreadsheets and hundreds of PDF
reports that require domain expertise to navigate.

## What You Could Build

Create a conversational assistant that handles one or both of these question types.

### Production Data Questions

Translate natural-language questions into queries over daily or monthly production data, including
oil, gas, and water rates and volumes per well over time.

Example: "Which well produced the most oil in 2014?"

A useful solution should make aggregations, units, date ranges, and well identifiers clear in its
answer.

### Document Questions

Use retrieval-based question answering over drilling, well, and geology reports. Answers should cite
the source documents and relevant pages so users can verify the result.

Example: "Summarize the geological setting of the Volve field."

## Data

The Volve dataset is free and openly licensed under the Equinor Open Data Licence. Rights to use the
data are already granted; no additional permission is needed.

- [Volve data sharing overview](https://www.equinor.com/energy/volve-data-sharing)
- Download access is provided through the Databricks Marketplace. Search for the `Equinor ASA`
  provider and follow the "How to get access" guide linked from the overview page.

Recommended starting slices:

- Per-well daily and monthly production volumes
- Drilling, well, and geology PDF reports

Large seismic and reservoir-model files are outside the recommended starting scope.

