do these two code changes. the purpose is to add an auto style preset that removes current random preset selection on auto. just do the code changes and verify if there is any impact of it that needs attention. 



DESIGN_PROMPTS = [
    {
        "name": "auto",
        "description": "Adaptive style that lets the LLM choose the best aesthetic based on user intent",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**ADAPTIVE DESIGN:**
Analyze the user's request and choose the most fitting visual style, color palette, and layout approach based on their content, industry, and intent. Use your best judgment for typography, spacing, color harmony, and visual hierarchy. Prioritize clarity, readability, and professional quality. There are no strict style constraints — deliver the design that best serves the user's goal.

**User Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "glassmorphism_premium",
        # ... rest of existing array unchanged



def get_design_prompt_template(theme: str) -> dict:
    """Get a design prompt template based on the specified theme."""
    # Get theme using name (works for "auto" and all named themes)
    for template in DESIGN_PROMPTS:
        if template["name"].lower() == theme.lower():
            logger.debug(f"Selected design prompt for theme '{theme}': {template['name']}")
            return template
    # If no match found, fallback to auto template
    logger.warning(f"No design prompt found for theme '{theme}', using auto")
    return DESIGN_PROMPTS[0]  # auto template is first in list