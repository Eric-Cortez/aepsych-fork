/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

// Define this so it can be easily modified in scripts (to host elsewhere)
// const baseUrl = "/"; // main aepsych branch
const baseUrl = "/aepsych-fork/"; // test branch


// List of projects/orgs using your project for the users page.
const users = [
  // {
  //   caption: 'AEPsych',
  //   // You will need to prepend the image path with your baseUrl
  //   // if it is not '/', like: '/test-site/img/image.jpg'.
  //   image: '/img/undraw_open_source.svg',
  //   infoLink: 'https://www.facebook.com',
  //   pinned: true,
  // },
];

const siteConfig = {
  title: 'AEPsych', // Title for your website.
  tagline: 'Adaptive experimentation for human perception and perceptually-informed outcomes',
  // url: 'https://facebookresearch.github.io', // Your website URL
  url: 'https://Eric-Cortez.github.io', // Your website URL
  baseUrl: baseUrl, // Base URL for your project */
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',
  cleanUrl: true, // No .html extensions for paths
  // Used for publishing and more
  // projectName: 'aepsych',
  // organizationName: 'facebookresearch',
  // Below for testing:
  projectName: 'aepsych-fork',
  organizationName: 'Eric-Cortez',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // NOINDEX - GOOGLE SEARCH web crawlers while testing website
  metadata: [
    {name: 'googlebot', content: 'noindex'},
    {name: 'robots', content: 'noindex'},
  ],
 // This would become <meta name="keywords" content="cooking, blog"> in the generated HTML
  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    {doc: 'introduction', label: 'Docs'},
    { href: `${baseUrl}tutorials/`, label: "Tutorials" },
    { href: `${baseUrl}api/`, label: "API Reference" },
    { href: "https://github.com/facebookresearch/aepsych", label: "GitHub" },
    // { search: false } disabled agnolia searchbar for now
  ],

  // If you have users set above, you add it here:
  users,

    // search integration w/ algolia
    // algolia: {
    //   apiKey: "",// will need to setup
    //   indexName: "aepsych" // will need to create and setup
    // },


  /* path to images for header/footer */
  // headerIcon: 'img/favicon.ico',
  footerIcon: 'img/favicon.ico',
  favicon: 'img/favicon.ico',

  /* Colors for website */
  colors: {
    primaryColor: '#20232a',
    secondaryColor: '#000080',
  },

  /* Custom fonts for website */
  /*
  fonts: {
    myFont: [
      "Times New Roman",
      "Serif"
    ],
    myOtherFont: [
      "-apple-system",
      "system-ui"
    ]
  },
  */

  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright © ${new Date().getFullYear()} Meta, Inc.`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'default',
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: [
     // Github buttons
    'https://buttons.github.io/buttons.js',
     // Copy-to-clipboard button for code blocks
     `${baseUrl}js/code_block_buttons.js`,
     "https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js",
      `${baseUrl}js/image_fail_to_load.js`,
       `${baseUrl}js/cookie_consent.js`,
  ],

  // CSS sources to load
  stylesheets: [
    `${baseUrl}static/css/basic.css`,
    `${baseUrl}static/css/custom.css`,
    `${baseUrl}static/css/code_block_buttons.css`,
    `${baseUrl}static/css/alabaster.css`
  ],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.

  // enable scroll to top button a the bottom of the site
  scrollToTop: true,

  // if true, expand/collapse links & subcategories in sidebar
  docsSideNavCollapsible: false,


  cleanUrl: true,
  // URL for editing docs - will link the edit button on the doc page once diff has landed
  editUrl: "https://github.com/facebookresearch/aepsych/blob/main/docs/",
  // Disable logo text so we can just show the logo
  // disableHeaderTitle: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/undraw_online.svg',
  twitterImage: 'img/undraw_tweetstorm.svg',

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  // docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  // repoUrl: 'https://github.com/facebook/test-site',

    // show html docs generated by sphinx
    wrapPagesHTML: true
};

module.exports = siteConfig;
