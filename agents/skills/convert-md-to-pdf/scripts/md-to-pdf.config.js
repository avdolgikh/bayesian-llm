module.exports = {
	body_class: 'markdown-body',
	stylesheet: [],
	css: `
		@import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;0,500;0,700;1,400&display=swap');
		@import url('https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap');

		body {
			font-family: Arial, 'Roboto', 'Segoe UI', 'Open Sans', 'Helvetica Neue', Helvetica, sans-serif;
			font-size: 0.8em;
		}

		h1, h2, h3, h4, h5, h6 {
			font-family: Arial, 'Roboto', 'Segoe UI', 'Open Sans', 'Helvetica Neue', Helvetica, sans-serif;
			font-weight: 400;
		}

		/* Tables: borders, header bg, striped rows */
		table {
			border-collapse: collapse;
			width: 100%;
			margin: 1em 0;
		}
		th, td {
			border: 1px solid #d0d7de;
			padding: 6px 12px;
			text-align: left;
		}
		th {
			background-color: #f0f3f6;
			font-weight: 600;
		}
		tr:nth-child(even) {
			background-color: #f8f9fa;
		}

		/* Horizontal rules */
		hr {
			border: none;
			border-top: 2px solid #d0d7de;
			margin: 2em 0;
		}

		/* Blockquotes (definitions box, callouts) */
		blockquote {
			border-left: 4px solid #d0d7de;
			background-color: #f6f8fa;
			padding: 0.75em 1em;
			margin: 1em 0;
		}

		/* Figure images: centered, reasonable width */
		img {
			max-width: 90%;
			height: auto;
			display: block;
			margin: 1em auto;
		}

		/* Figure captions (alt text rendered as figcaption by some renderers) */
		figcaption, .caption {
			text-align: center;
			font-size: 0.9em;
			color: #555;
			margin-top: 0.5em;
		}

		/* MathJax display math spacing */
		mjx-container[display="true"] {
			margin: 0.75em 0 !important;
		}
	`,
	marked_extensions: [
		{
			// Protect $$...$$ blocks from markdown underscore processing
			extensions: [
				{
					name: 'math_block',
					level: 'block',
					start(src) { return src.match(/^\$\$/m)?.index; },
					tokenizer(src) {
						const match = src.match(/^\$\$([\s\S]*?)\$\$/);
						if (match) {
							return { type: 'html', raw: match[0], text: match[0] };
						}
					},
				},
			],
		},
	],
	script: [
		{
			// MathJax config — must come before MathJax itself
			content: `
				window.MathJax = {
					tex: {
						inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
						displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
						processEscapes: true
					},
					options: {
						skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
					},
					startup: {
						pageReady: () => {
							return MathJax.startup.defaultPageReady().then(() => {
								document.body.setAttribute('data-mathjax-done', 'true');
							});
						}
					}
				};
			`,
		},
		{ url: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js' },
	],
	launch_options: {
		args: ['--no-sandbox'],
	},
	pdf_options: {
		format: 'A4',
		margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' },
		printBackground: true,
	},
};
