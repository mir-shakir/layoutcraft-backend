# Design Prompt Templates for Visual Asset Generation
# Each template focuses on different design aesthetics while maintaining high quality standards

# Common instructions shared across all templates
COMMON_INSTRUCTIONS = """
**CRITICAL OUTPUT REQUIREMENTS:**
- Your response must ONLY contain raw HTML code starting with `<!DOCTYPE html>` and ending with `</html>`
- NO explanations, NO markdown fences, NO additional text outside HTML
- ALL styling must be within a single `<style>` tag in the HTML head
- Container dimensions must be exactly 1200x630px
- No external files, no JavaScript, no CDN links allowed
- Only Font Awesome icons are allowed, and the cdn link to use for that is "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/7.0.0/css/all.min.css". you can use it in the HTML.
- For Fonts Use standard fonts from google fonts and import only the ones you need. Add the required font import in the HTML head.

**QUALITY STANDARDS:**
- Aim for EXCEPTIONAL, award-winning design quality
- Create designs worthy of premium design portfolios
- Every element must be purposeful and professionally crafted
- Use modern design principles and contemporary aesthetics
- Maximize use of Font Awesome icons to fulfil user requests

**USER REQUIREMENTS COMPLIANCE:**
- STRICTLY follow any specific requirements from the user prompt
- If user specifies "no text" - create purely visual designs
- If user mentions specific colors - incorporate them as primary palette
- If user requests specific style - prioritize that over template defaults
- Respect any layout, size, or content specifications mentioned
- Avoid adding any made-up content or generic placeholders
- If user prompt is vague, use best judgment to create a high-quality design

**TEXT HANDLING:**
- Only include text that adds real value to the design
- Avoid placeholder text like "Lorem ipsum" or generic phrases
- If text is needed, make it contextually relevant to the user's request
- Keep text minimal, impactful, and professionally styled
- If user prompt suggests avoiding text, create purely visual compositions

**TECHNICAL EXCELLENCE:**
- Use CSS Grid, Flexbox for modern layouts
- Implement proper color theory and contrast
- Apply sophisticated typography hierarchy when text is appropriate
- Ensure responsive design principles within fixed dimensions
- Optimize for visual impact and professional presentation
- Most Important: Use div with .container with exact dimensions and no shadows to contain all content and facilitate easy screenshots capture. there should be no other element with this class.
"""

DESIGN_PROMPTS = [
    {
        "name": "glassmorphism_premium",
        "description": "Modern glassmorphism with depth and sophistication",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DESIGN AESTHETIC: Premium Glassmorphism**
Create a sophisticated glassmorphism design that exudes premium quality and modern elegance.

**BACKGROUND CONSTRUCTION:**
- Multi-layered gradient background using 3-4 complementary colors
- Use `radial-gradient` and `linear-gradient` combinations for depth
- Colors should be muted, sophisticated (think premium brand palettes)
- Add subtle geometric shapes with very low opacity (0.05-0.15)
- Implement floating elements with different blur levels for depth

**GLASSMORPHISM ELEMENTS:**
- Primary content area: `background: rgba(255, 255, 255, 0.1)` to `rgba(255, 255, 255, 0.2)`
- Essential: `backdrop-filter: blur(20px)` for frosted glass effect
- Borders: `border: 1px solid rgba(255, 255, 255, 0.2)`
- Multiple glass layers with varying transparency levels
- Subtle inner shadows: `box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1)`

**VISUAL HIERARCHY:**
- Create clear focal points using transparency variations
- Use scale and positioning to guide viewer attention
- Implement subtle animations via CSS transforms if appropriate
- Layer elements to create visual depth and interest

**PURPOSE:** Create poster-quality designs for tech brands, modern businesses, or premium products.

**User's Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "bold_geometric_solid",
        "description": "Bold geometric design with solid colors and strong shapes",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DESIGN AESTHETIC: Bold Geometric with Solid Colors**
Create a striking, contemporary design using bold geometric shapes and confident solid color combinations.

**COLOR STRATEGY:**
- Use 2-3 bold, complementary solid colors maximum
- Consider modern palettes: deep blues with vibrant oranges, rich purples with bright yellows
- Implement 60-30-10 color rule for balance
- Use pure, saturated colors for maximum impact
- Add one neutral (white, black, or gray) for breathing space

**GEOMETRIC COMPOSITION:**
- Large, confident geometric shapes (circles, triangles, rectangles, hexagons)
- Overlapping elements with different opacity levels
- Strong diagonal compositions for dynamic energy
- Use negative space strategically for balance
- Implement golden ratio or rule of thirds for proportions

**LAYOUT PRINCIPLES:**
- Asymmetrical balance for modern appeal
- Clear visual hierarchy through size and color
- Bold typography when text is required (modern sans-serif)
- Sharp, clean edges with occasional rounded corners for softness
- Strong contrast between elements for clarity

**VISUAL IMPACT:**
- Aim for designs that would work as large-format prints
- Create immediate visual impact that stops scrolling
- Use scale dramatically - some elements very large, others very small
- Implement depth through color saturation and overlap

**PURPOSE:** Perfect for startup brands, creative agencies, event posters, or bold marketing materials.

**User's Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "textured_organic_patterns",
        "description": "Organic patterns with subtle textures and natural flow",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DESIGN AESTHETIC: Organic Textured Patterns**
Create a sophisticated design using organic shapes, subtle textures, and natural flow patterns.

**ORGANIC ELEMENTS:**
- Fluid, curved shapes inspired by nature (waves, leaves, clouds, topography)
- Avoid hard geometric edges - everything should flow naturally
- Use CSS `border-radius` creatively for organic shapes
- Implement wave patterns using CSS clip-path or border-radius
- Layer organic shapes with varying opacity and size

**TEXTURE IMPLEMENTATION:**
- CSS gradients to simulate paper, fabric, or natural textures
- Subtle noise effects using multiple small radial gradients
- Layered transparency to create depth and texture
- Use box-shadow creatively to add dimension
- Implement subtle pattern overlays with low opacity

**COLOR PALETTE:**
- Earth tones and natural colors (terracotta, sage, cream, warm grays)
- Or modern interpretations (dusty blues, muted corals, soft lavenders)
- Maximum 4 colors in harmonious relationship
- Use color temperature to create mood and depth
- Implement subtle color transitions and blending

**COMPOSITION FLOW:**
- S-curves and natural movement patterns
- Elements should guide the eye in organic paths
- Use scale variations mimicking natural growth patterns
- Implement breathing room - let elements flow naturally
- Balance density and openness like natural landscapes

**PURPOSE:** Ideal for wellness brands, artisanal products, lifestyle content, or sophisticated editorial designs.

**User's Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "minimal_luxury_space",
        "description": "Ultra-minimal luxury design with sophisticated use of space",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DESIGN AESTHETIC: Luxury Minimalism**
Create an ultra-sophisticated minimal design that communicates premium quality through restraint and perfect execution.

**MINIMALIST PRINCIPLES:**
- Maximum 2 colors plus one neutral (white, black, or warm gray)
- Generous negative space - let elements breathe
- Perfect typography hierarchy when text is essential
- Single focal point with supporting elements only
- Every element must earn its place - remove anything unnecessary

**LUXURY INDICATORS:**
- Subtle, expensive-feeling details (thin lines, perfect spacing)
- Premium color combinations (deep navy + gold, charcoal + cream)
- High contrast for clarity and sophistication
- Subtle gradients or shadows for depth without complexity
- Perfect symmetry or intentional asymmetry - nothing accidental

**SPATIAL COMPOSITION:**
- Use golden ratio for proportions
- Implement rule of thirds for placement
- Create visual breathing space around key elements
- Use scale dramatically - one large element, everything else much smaller
- Perfect alignment and consistent spacing throughout

**TYPOGRAPHIC EXCELLENCE:**
- If text is required, use maximum 2 font weights
- Perfect letter spacing and line height
- Text as design element, not just information
- Consider text as texture and visual element
- Hierarchy through size and weight, not color

**MATERIAL FEEL:**
- Suggest premium materials through color and shadow
- Subtle depth without complexity
- Clean edges with occasional soft shadows
- Use transparency sparingly but effectively

**PURPOSE:** Perfect for luxury brands, high-end services, premium products, or sophisticated corporate materials.

**User's Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "vibrant_gradient_energy",
        "description": "Dynamic gradients with vibrant energy and movement",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DESIGN AESTHETIC: Vibrant Gradient Energy**
Create an energetic, contemporary design using dynamic gradients and vibrant color combinations that pulse with life.

**GRADIENT MASTERY:**
- Multi-stop gradients with 3-5 vibrant colors
- Combine linear and radial gradients for complexity
- Use gradient mesh effects through overlapping gradients
- Implement directional gradients (45°, 135° angles) for movement
- Layer gradients with different blend modes (multiply, overlay, screen)

**COLOR VIBRANCE:**
- Bold, saturated color combinations (electric blues, hot pinks, bright yellows)
- Or sophisticated vibrant pairs (deep purples + bright corals)
- Use color psychology - warm colors for energy, cool for tech
- Implement smooth color transitions that create visual flow
- Consider trendy color combinations from current design movements

**DYNAMIC MOVEMENT:**
- Diagonal compositions for energy and movement
- Flowing shapes that suggest motion and progression
- Use CSS transforms for subtle rotation and skewing
- Implement curves and waves that create visual rhythm
- Scale elements to suggest depth and movement toward viewer

**VISUAL ENERGY:**
- High contrast between light and dark areas
- Bright highlights and rich shadows
- Elements that seem to glow or pulse with energy
- Use negative space to make vibrant areas pop
- Create focal points through color intensity

**MODERN TECHNIQUE:**
- Implement CSS `filter` effects (brightness, contrast, saturation)
- Use `mix-blend-mode` for sophisticated color interactions
- Layer transparent elements for color mixing effects
- Create depth through gradient layering and transparency

**PURPOSE:** Perfect for tech startups, creative agencies, music/entertainment, fitness brands, or any content requiring high energy and modern appeal.

**User's Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "dark_neon_tech",
        "description": "Dark theme with neon accents and futuristic tech aesthetic",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DESIGN AESTHETIC: Dark Tech with Neon**
Create a futuristic, high-tech design using dark themes with strategic neon accents and sci-fi inspired elements.

**DARK FOUNDATION:**
- Deep background colors (rich blacks, dark navies, charcoal)
- Use gradients from black to very dark colors for depth
- Implement subtle texture through dark color variations
- Create depth through layered dark tones
- Never use pure black - always rich, complex darks

**NEON ACCENT STRATEGY:**
- 1-2 bright neon colors maximum (electric blue, hot pink, bright cyan, lime green)
- Use neon colors sparingly for maximum impact
- Implement CSS `box-shadow` with neon colors for glow effects
- Create light trails and glowing borders
- Use neon colors for highlighting key elements only

**TECH ELEMENTS:**
- Grid patterns with low opacity for tech aesthetic
- Subtle geometric patterns suggesting circuit boards or data
- Clean lines and angular shapes
- Hexagonal or other tech-inspired geometric elements
- Implement subtle scan lines or digital noise effects

**FUTURISTIC COMPOSITION:**
- Angular, precise layouts with mathematical proportions
- Use negative space to create clean, high-tech feel
- Implement subtle animations through CSS transforms
- Create depth through layering and transparency
- Sharp, clean edges with occasional strategic rounding

**LIGHTING EFFECTS:**
- Strategic use of CSS `box-shadow` for glow and depth
- Inner shadows to create recessed elements
- Highlight edges with thin, bright lines
- Use gradients to simulate lighting and reflection
- Create focal points through strategic illumination

**PURPOSE:** Ideal for tech companies, gaming brands, cybersecurity, AI/ML products, or any futuristic/sci-fi themed content.

**User's Request:** "{{user_prompt}}"
"""
    },
    {
        "name": "editorial_magazine_layout",
        "description": "Editorial-style layout with sophisticated typography and visual hierarchy",
        "prompt": f"""{COMMON_INSTRUCTIONS}

**DESIGN AESTHETIC: Editorial Magazine Quality**
Create a sophisticated editorial design that rivals premium magazine layouts with impeccable typography and visual storytelling.

**EDITORIAL PRINCIPLES:**
- Clear information hierarchy through typography and spacing
- Grid-based layout system for professional consistency
- Strategic use of whitespace for readability and elegance
- Balance between visual elements and content organization
- Editorial color palette - sophisticated and purposeful

**TYPOGRAPHY MASTERY:**
- Maximum 2 font families - one serif, one sans-serif if text is required
- Create dramatic scale contrasts between headline and body text
- Perfect line height, letter spacing, and paragraph spacing
- Use typography as a visual design element
- Implement drop caps, pull quotes, or other editorial techniques when appropriate

**LAYOUT SOPHISTICATION:**
- Multi-column layouts when appropriate
- Strategic image placement and text wrapping
- Use of margins and gutters like premium publications
- Implement visual anchors and flow guides
- Balance text density with visual breathing space

**VISUAL STORYTELLING:**
- Create narrative flow through visual hierarchy
- Use color and scale to guide reader attention
- Implement subtle visual cues for content organization
- Balance information density with visual appeal
- Consider the user's content as editorial subject matter

**PREMIUM DETAILS:**
- Subtle rules, borders, and dividers for organization
- Sophisticated color palette (maximum 3-4 colors)
- Perfect alignment and consistent spacing throughout
- Use of premium design elements (subtle shadows, refined borders)
- Attention to micro-typography and spacing details

**PURPOSE:** Perfect for blogs, magazines, editorial content, professional reports, or any content-focused design requiring sophistication.

**User's Request:** "{{user_prompt}}"
"""
    }
]

# Function to get template by name (useful for debugging or specific selection)
def get_template_by_name(name: str) -> dict:
    """Get a specific template by name."""
    for template in DESIGN_PROMPTS:
        if template["name"] == name:
            return template
    return DESIGN_PROMPTS[0]  # Return first template as fallback

# Function to get all template names
def get_all_template_names() -> list:
    """Get list of all available template names."""
    return [template["name"] for template in DESIGN_PROMPTS]