//! `#[tool]` — turn an async function into a `harness_core::Tool`.
//!
//! ```ignore
//! /// Read a UTF-8 text file and return its contents.
//! #[tool]
//! async fn read_file(path: String, ctx: &ToolContext) -> Result<String, ToolError> { ... }
//!
//! /// Run a shell command.
//! #[tool(approval)] // runtime pauses for human approval before executing
//! async fn bash(command: String) -> Result<String, ToolError> { ... }
//! ```
//!
//! The macro derives the tool's JSON Schema from the parameter types at
//! compile time (via `schemars`), uses the doc comment as the description,
//! and replaces the function with a unit struct of the same name implementing
//! `Tool`. A parameter of type `&ToolContext` (any position) is injected by
//! the runtime rather than exposed in the schema. `Option<T>` parameters are
//! optional in the schema and may be omitted by the model.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, spanned::Spanned, FnArg, ItemFn, Pat, Type};

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let needs_approval = match parse_attr(attr) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };
    let func = parse_macro_input!(item as ItemFn);
    match expand(func, needs_approval) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn parse_attr(attr: TokenStream) -> syn::Result<bool> {
    if attr.is_empty() {
        return Ok(false);
    }
    let ident: syn::Ident = syn::parse(attr)?;
    if ident == "approval" {
        Ok(true)
    } else {
        Err(syn::Error::new(
            ident.span(),
            "unknown #[tool(...)] option; supported: `approval`",
        ))
    }
}

fn expand(func: ItemFn, needs_approval: bool) -> syn::Result<TokenStream2> {
    if func.sig.asyncness.is_none() {
        return Err(syn::Error::new(
            func.sig.span(),
            "#[tool] functions must be async",
        ));
    }

    let vis = &func.vis;
    let fn_name = &func.sig.ident;
    let name_str = fn_name.to_string();
    let description = doc_comment(&func);

    // Split parameters into schema-visible inputs and the injected context.
    let mut fields = Vec::new(); // struct fields for the input schema
    let mut call_args = Vec::new(); // arguments in original order
    for arg in &func.sig.inputs {
        let FnArg::Typed(pat_ty) = arg else {
            return Err(syn::Error::new(
                arg.span(),
                "#[tool] does not support `self`",
            ));
        };
        let Pat::Ident(pat_ident) = pat_ty.pat.as_ref() else {
            return Err(syn::Error::new(
                pat_ty.pat.span(),
                "#[tool] parameters must be plain identifiers",
            ));
        };
        let ident = &pat_ident.ident;
        let ty = pat_ty.ty.as_ref();

        if is_tool_context(ty) {
            call_args.push(quote!(__ctx));
        } else {
            let default_attr = if is_option(ty) {
                quote!(#[serde(default)])
            } else {
                quote!()
            };
            fields.push(quote! {
                #default_attr
                #ident: #ty
            });
            call_args.push(quote!(__input.#ident));
            let _ = ident;
        }
    }

    let approval = needs_approval;
    let input_struct = format_ident!("__{}Input", fn_name);

    Ok(quote! {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        #vis struct #fn_name;

        const _: () = {
            #[derive(
                ::harness_core::__private::serde::Deserialize,
                ::harness_core::__private::schemars::JsonSchema,
            )]
            #[serde(crate = "::harness_core::__private::serde")]
            #[schemars(crate = "::harness_core::__private::schemars")]
            #[allow(non_camel_case_types)]
            struct #input_struct {
                #(#fields,)*
            }

            // The original function, kept intact so its body type-checks with
            // normal spans and error messages.
            #func

            #[::harness_core::async_trait]
            impl ::harness_core::Tool for #fn_name {
                fn name(&self) -> &str {
                    #name_str
                }

                fn description(&self) -> &str {
                    #description
                }

                fn input_schema(&self) -> ::harness_core::__private::serde_json::Value {
                    let schema = ::harness_core::__private::schemars::schema_for!(#input_struct);
                    ::harness_core::__private::serde_json::to_value(schema)
                        .expect("tool input schema must serialize")
                }

                fn needs_approval(&self) -> bool {
                    #approval
                }

                async fn call(
                    &self,
                    input: ::harness_core::__private::serde_json::Value,
                    __ctx: &::harness_core::ToolContext,
                ) -> ::std::result::Result<::harness_core::ToolOutput, ::harness_core::ToolError> {
                    #[allow(unused_variables)]
                    let __input: #input_struct =
                        ::harness_core::__private::serde_json::from_value(input)
                            .map_err(|e| ::harness_core::ToolError::InvalidInput(e.to_string()))?;
                    let out = #fn_name(#(#call_args),*).await?;
                    ::std::result::Result::Ok(::harness_core::IntoToolOutput::into_tool_output(out))
                }
            }
        };
    })
}

fn doc_comment(func: &ItemFn) -> String {
    let lines: Vec<String> = func
        .attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            match &attr.meta {
                syn::Meta::NameValue(nv) => match &nv.value {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) => Some(s.value().trim().to_string()),
                    _ => None,
                },
                _ => None,
            }
        })
        .collect();
    lines.join("\n")
}

fn is_tool_context(ty: &Type) -> bool {
    let Type::Reference(r) = ty else { return false };
    let Type::Path(p) = r.elem.as_ref() else {
        return false;
    };
    p.path
        .segments
        .last()
        .map(|s| s.ident == "ToolContext")
        .unwrap_or(false)
}

fn is_option(ty: &Type) -> bool {
    let Type::Path(p) = ty else { return false };
    p.path
        .segments
        .last()
        .map(|s| s.ident == "Option")
        .unwrap_or(false)
}
