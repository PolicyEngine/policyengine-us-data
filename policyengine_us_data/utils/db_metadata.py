"""
Utility functions for managing database metadata (sources, variable groups, etc.)
"""

from typing import Optional
from sqlmodel import Session, select
from policyengine_us_data.db.create_database_tables import (
    Source,
    SourceType,
    VariableGroup,
    VariableMetadata,
)


def get_or_create_source(
    session: Session,
    name: str,
    source_type: SourceType,
    vintage: Optional[str] = None,
    description: Optional[str] = None,
    url: Optional[str] = None,
    notes: Optional[str] = None,
) -> Source:
    """
    Get an existing source or create a new one.
    
    Args:
        session: Database session
        name: Name of the data source
        source_type: Type of source (administrative, survey, etc.)
        vintage: Version or year of the data
        description: Detailed description
        url: Reference URL
        notes: Additional notes
        
    Returns:
        Source object with source_id populated
    """
    # Try to find existing source by name and vintage
    query = select(Source).where(Source.name == name)
    if vintage:
        query = query.where(Source.vintage == vintage)
    
    source = session.exec(query).first()
    
    if not source:
        # Create new source
        source = Source(
            name=name,
            type=source_type,
            vintage=vintage,
            description=description,
            url=url,
            notes=notes,
        )
        session.add(source)
        session.flush()  # Get the auto-generated ID
    
    return source


def get_or_create_variable_group(
    session: Session,
    name: str,
    category: str,
    is_histogram: bool = False,
    is_exclusive: bool = False,
    aggregation_method: Optional[str] = None,
    display_order: Optional[int] = None,
    description: Optional[str] = None,
) -> VariableGroup:
    """
    Get an existing variable group or create a new one.
    
    Args:
        session: Database session
        name: Unique name of the variable group
        category: High-level category (demographic, benefit, tax, income)
        is_histogram: Whether this represents a distribution
        is_exclusive: Whether variables are mutually exclusive
        aggregation_method: How to aggregate (sum, weighted_avg, etc.)
        display_order: Order for display
        description: Description of the group
        
    Returns:
        VariableGroup object with group_id populated
    """
    group = session.exec(
        select(VariableGroup).where(VariableGroup.name == name)
    ).first()
    
    if not group:
        group = VariableGroup(
            name=name,
            category=category,
            is_histogram=is_histogram,
            is_exclusive=is_exclusive,
            aggregation_method=aggregation_method,
            display_order=display_order,
            description=description,
        )
        session.add(group)
        session.flush()  # Get the auto-generated ID
    
    return group


def get_or_create_variable_metadata(
    session: Session,
    variable: str,
    group: Optional[VariableGroup] = None,
    display_name: Optional[str] = None,
    display_order: Optional[int] = None,
    units: Optional[str] = None,
    is_primary: bool = True,
    notes: Optional[str] = None,
) -> VariableMetadata:
    """
    Get existing variable metadata or create new.
    
    Args:
        session: Database session
        variable: PolicyEngine variable name
        group: Variable group this belongs to
        display_name: Human-readable name
        display_order: Order within group
        units: Units of measurement
        is_primary: Whether this is a primary variable
        notes: Additional notes
        
    Returns:
        VariableMetadata object
    """
    metadata = session.exec(
        select(VariableMetadata).where(VariableMetadata.variable == variable)
    ).first()
    
    if not metadata:
        metadata = VariableMetadata(
            variable=variable,
            group_id=group.group_id if group else None,
            display_name=display_name or variable,
            display_order=display_order,
            units=units,
            is_primary=is_primary,
            notes=notes,
        )
        session.add(metadata)
        session.flush()
    
    return metadata