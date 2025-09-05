# Design Prompt Templates for Visual Asset Generation
# Optimized for token efficiency while maintaining design quality

COMMON_INSTRUCTIONS = """
**OUTPUT FORMAT:**
- Raw HTML only: `<!DOCTYPE html>` to `</html>` - NO explanations/markdown
- All CSS in single `<style>` tag in head
- Use `.container` class with exact user-specified dimensions, no shadows
- Font Awesome CDN: "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/7.0.0/css/all.min.css"
- use a combinatioin of custom css and  tailwind to make it less verbose "https://cdn.tailwindcss.com".
- Google Fonts imports in head as needed

**REQUIREMENTS:**
- STRICTLY follow ALL user specifications (colors, style, layout, dimensions, text preferences)
- Award-winning, portfolio-level design quality
- Modern aesthetics with purposeful elements
- CSS Grid/Flexbox, proper color theory
- Maximize Font Awesome icon usage
- No external files/JavaScript (except specified CDNs)
- Minimal contextual text only - avoid Lorem ipsum/placeholders
- If "no text" specified → purely visual design
"""

DESIGN_PROMPTS = [
    {
        "name": "glassmorphism_premium",
        "description": "Modern glassmorphism with depth and sophistication",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**GLASSMORPHISM AESTHETIC:**
Multi-layered gradient backgrounds (3-4 sophisticated colors), frosted glass elements with `backdrop-filter: blur(20px)`, `background: rgba(255,255,255,0.1-0.2)`, subtle borders `rgba(255,255,255,0.2)`. Layer transparent elements for depth. Perfect for tech/premium brands.

**User Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "bold_geometric_solid", 
        "description": "Bold geometric design with solid colors and strong shapes",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**BOLD GEOMETRIC AESTHETIC:**
2-3 bold complementary solid colors, large confident geometric shapes (circles, triangles, hexagons), asymmetrical balance, strong diagonals, strategic negative space. Use 60-30-10 color rule, overlapping elements with opacity. Perfect for startups/creative agencies.

**User Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "textured_organic_patterns",
        "description": "Organic patterns with subtle textures and natural flow", 
        "prompt": f"""{COMMON_INSTRUCTIONS}

**ORGANIC TEXTURED AESTHETIC:**
Fluid curved shapes, natural flow patterns, earth tones/modern naturals (max 4 colors), CSS gradients for texture effects, layered transparency for depth. S-curves, organic movement, breathing space. Perfect for wellness/lifestyle brands.

**User Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "minimal_luxury_space",
        "description": "Ultra-minimal luxury design with sophisticated use of space",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**LUXURY MINIMAL AESTHETIC:**
Max 2 colors + neutral, generous negative space, single focal point, perfect typography hierarchy, golden ratio proportions, subtle premium details (thin lines, perfect spacing), high contrast. Every element must earn its place. Perfect for luxury brands.

**User Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "vibrant_gradient_energy",
        "description": "Dynamic gradients with vibrant energy and movement",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**VIBRANT GRADIENT AESTHETIC:**
Multi-stop gradients (3-5 vibrant colors), combine linear/radial gradients, diagonal compositions for movement, high contrast light/dark, CSS blend modes, flowing shapes suggesting motion. Perfect for tech startups/creative agencies.

**User Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "dark_neon_tech",
        "description": "Dark theme with neon accents and futuristic tech aesthetic", 
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DARK TECH AESTHETIC:**
Deep backgrounds (rich blacks, dark navies), 1-2 neon accent colors max, CSS glow effects with `box-shadow`, grid patterns, angular layouts, tech geometric elements, strategic lighting effects. Perfect for tech/gaming/AI brands.

**User Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "editorial_magazine_layout",
        "description": "Editorial-style layout with sophisticated typography and visual hierarchy",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**EDITORIAL AESTHETIC:**
Grid-based layouts, typography hierarchy (max 2 fonts), multi-column when appropriate, strategic whitespace, sophisticated 3-4 color palette, visual storytelling flow, premium spacing/alignment. Perfect for content-focused designs.

**User Request:** "{{user_prompt}}"
"""
    }
]

# Optimized edit prompt
HTML_EDIT_PROMPT = """
Expert designer: modify provided HTML based on user request. 
Context: This html is converted into png using a headless browser to create a design asset. try to to follow the design principles while making the changes.

**RULES:**
1. MINIMAL CHANGES: Preserve original design/layout/styles. Only modify what's absolutely necessary.
2. CODE-ONLY OUTPUT: Complete raw HTML from `<!DOCTYPE html>` to `</html>`. NO explanations/markdown.
3. VALID & SELF-CONTAINED: All styles in single `<style>` tag.

**ORIGINAL HTML:**
```html
{original_html}
```
**EDIT REQUEST:** "{edit_prompt}"
"""

def get_template_by_name(name: str) -> dict:
    """Get specific template by name."""
    return next((t for t in DESIGN_PROMPTS if t["name"] == name), DESIGN_PROMPTS[0])

def get_all_template_names() -> list:
    """Get all available template names."""
    return [template["name"] for template in DESIGN_PROMPTS]