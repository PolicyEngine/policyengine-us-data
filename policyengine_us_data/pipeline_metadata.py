"""Pipeline metadata decorators.

Attach pipeline documentation to functions without changing their behavior.
The Griffe extraction script reads these decorators statically (via AST)
to generate pipeline.json for visualization.

Usage:
    from policyengine_us_data.pipeline_metadata import pipeline_node
    from policyengine_us_data.pipeline_schema import PipelineNode

    @pipeline_node(PipelineNode(
        id="add_rent",
        label="Rent Imputation (QRF)",
        node_type="process",
        description="Impute rent and real estate taxes using QRF from ACS",
    ))
    def add_rent(self, cps, person, household):
        ...
"""

from policyengine_us_data.pipeline_schema import PipelineNode


def pipeline_node(node: PipelineNode):
    """Decorator that attaches pipeline node metadata to a function.

    Does not modify the function's behavior. The metadata is stored
    as func._pipeline_node and read by the Griffe extension during
    static analysis.

    Args:
        node: A PipelineNode describing this function's role in the
            pipeline — its type, description, and visual properties.
    """

    def wrapper(func):
        func._pipeline_node = node
        return func

    return wrapper
